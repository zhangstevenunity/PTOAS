//===---------------- InsertLoadStoreForMixCV.cpp -------------------------===//
//
// cube_matmul_vadd_2d rewrite:
//
// - add args: %arg_ws: memref<?xi8,  #pto.address_space<gm>>
//           : %arg_ffts: memref<?xi64, #pto.address_space<gm>>
// - insert: pto.set_ffts %arg_ffts
// - force tile_idx_x/y -> 0 and erase them
// - replace tail:
//     (CC -> CBUF) + addf(CBUF) + store(GM)
//   with:
//     CC -> workspace(GM view from %arg_ws) -> UB
//     C(GM) -> UB
//     addf(UB)
//     store(GM)
//
// Notes:
// - Do NOT use AsmParser / parseAttribute.
// - Avoid memref_ext.alloc_workspace (unregistered dialect crash) by using memref.view.
// - PTO generated builders: prefer the shortest builder to avoid optional-arg mismatch.
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"   // IRRewriter
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
namespace pto {

namespace {

// ------------------------- helpers -------------------------

static bool memSpaceContains(Type t, StringRef needle) {
  auto mr = dyn_cast<MemRefType>(t);
  if (!mr)
    return false;
  Attribute ms = mr.getMemorySpace();
  if (!ms)
    return false;
  std::string s;
  llvm::raw_string_ostream os(s);
  ms.print(os);
  os.flush();
  return StringRef(s).contains(needle);
}

static arith::ConstantIndexOp getOrCreateC0(IRRewriter &rewriter,
                                            func::FuncOp f) {
  Block &entry = f.front();
  for (Operation &op : entry.getOperations()) {
    if (auto cst = dyn_cast<arith::ConstantIndexOp>(op)) {
      if (cst.value() == 0)
        return cst;
    }
  }
  rewriter.setInsertionPointToStart(&entry);
  return rewriter.create<arith::ConstantIndexOp>(f.getLoc(), 0);
}

static memref::AllocOp createAllocWithAlign(IRRewriter &rewriter, Location loc,
                                            MemRefType ty, int64_t align = 64) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, ty);
  alloc->setAttr("alignment", rewriter.getI64IntegerAttr(align));
  return alloc;
}

// ------------------------- matching -------------------------
//
// Match old tail pattern:
//
//   %mm_cc = pto.matmul_dps ... outs(%mm_cc : ...cc...)
//   %mm_cbuf = (pto.load_dps/pto.copy/pto.mov) ins(%mm_cc) ... -> ...cbuf...
//   %c_cbuf = pto.load_dps ins(%c_gm) outs(%c_cbuf)
//   %out_cbuf = pto.addf_dps ins(%c_cbuf, %mm_cbuf) outs(%out_cbuf)
//   pto.store_dps ins(%out_cbuf) outs(%o_gm)
//
// and capture %c_gm and %o_gm.
//

struct MatchInfo {
  Operation *matmulDps = nullptr;
  Value mmCCBuf;                 // ...cc...

  Operation *ccToVec = nullptr;  // CC->CBUF
  Value mmVecBuf;                // ...cbuf...

  Operation *cGmToCbuf = nullptr;
  Value cGMSubview;              // ...gm...
  Value cCbufBuf;                // ...cbuf...

  Operation *addfCbuf = nullptr;
  Value outCbufBuf;              // ...cbuf...

  Operation *storeToOut = nullptr;
  Value outGMSubview;            // ...gm...
};

static bool match(func::FuncOp func, MatchInfo &mi) {
  if (func.getNumArguments() != 11)
    return false;

  // 1) find matmul_dps and dst(CC)
  for (Operation &op : func.front().getOperations()) {
    if (op.getName().getStringRef() != "pto.tmatmul")
      continue;
    if (op.getNumOperands() < 1)
      continue;
    Value dst = op.getOperand(op.getNumOperands() - 1);
    if (!memSpaceContains(dst.getType(), "cc"))
      continue;
    mi.matmulDps = &op;
    mi.mmCCBuf = dst;
    break;
  }
  if (!mi.matmulDps || !mi.mmCCBuf)
    return false;

  // 2) find final store_dps (CBUF -> GM)
  for (Operation &op : func.front().getOperations()) {
    if (op.getName().getStringRef() != "pto.tstore")
      continue;
    if (op.getNumOperands() < 2)
      continue;
    Value src = op.getOperand(0);
    Value dst = op.getOperand(1);
    if (memSpaceContains(src.getType(), "cbuf") && memSpaceContains(dst.getType(), "gm")) {
      mi.storeToOut = &op;
      mi.outGMSubview = dst;
      break;
    }
  }
  if (!mi.storeToOut)
    return false;

  // 3) addf_dps should define store src
  Value storeSrc = mi.storeToOut->getOperand(0);
  Operation *addfDef = storeSrc.getDefiningOp();
  if (!addfDef || addfDef->getName().getStringRef() != "pto.addf_dps")
    return false;
  mi.addfCbuf = addfDef;

  if (mi.addfCbuf->getNumOperands() < 3)
    return false;
  mi.outCbufBuf = mi.addfCbuf->getOperand(mi.addfCbuf->getNumOperands() - 1);

  Value addIn0 = mi.addfCbuf->getOperand(0);
  Value addIn1 = mi.addfCbuf->getOperand(1);
  if (!(memSpaceContains(addIn0.getType(), "cbuf") &&
        memSpaceContains(addIn1.getType(), "cbuf")))
    return false;

  // 4) locate CC->CBUF op producing one add input
  auto isCCtoCBUF = [&](Operation *op, Value &dstOut) -> bool {
    StringRef n = op->getName().getStringRef();
    if (n != "pto.tload" && n != "pto.tmov" && n != "pto.mov")
      return false;
    if (op->getNumOperands() < 2)
      return false;

    Value src = op->getOperand(0);
    Value dst = op->getOperand(op->getNumOperands() - 1);

    if (!memSpaceContains(src.getType(), "cc"))
      return false;
    if (!memSpaceContains(dst.getType(), "cbuf"))
      return false;

    dstOut = dst;
    return true;
  };

  Value mmVec;
  Operation *cc2vec = nullptr;
  for (Operation &op : func.front().getOperations()) {
    Value dst;
    if (!isCCtoCBUF(&op, dst))
      continue;
    if (op.getOperand(0) != mi.mmCCBuf)
      continue;
    if (dst == addIn0 || dst == addIn1) {
      cc2vec = &op;
      mmVec = dst;
      break;
    }
  }
  if (!cc2vec || !mmVec)
    return false;

  mi.ccToVec = cc2vec;
  mi.mmVecBuf = mmVec;

  // the other add input is C(cbuf)
  mi.cCbufBuf = (addIn0 == mmVec) ? addIn1 : addIn0;

  // 5) find GM->CBUF load for C
  for (Operation &op : func.front().getOperations()) {
    if (op.getName().getStringRef() != "pto.tload")
      continue;
    if (op.getNumOperands() < 2)
      continue;
    Value src = op.getOperand(0);
    Value dst = op.getOperand(1);
    if (dst != mi.cCbufBuf)
      continue;
    if (!memSpaceContains(src.getType(), "gm"))
      continue;
    if (!memSpaceContains(dst.getType(), "cbuf"))
      continue;
    mi.cGmToCbuf = &op;
    mi.cGMSubview = src;
    break;
  }
  if (!mi.cGmToCbuf || !mi.cGMSubview)
    return false;

  return true;
}

// ------------------------- pass -------------------------

struct InsertLoadStoreForMixCVPass
    : public PassWrapper<InsertLoadStoreForMixCVPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertLoadStoreForMixCVPass)

  StringRef getArgument() const override {
    return "pto-insert-load-store-for-mix-cv";
  }
  StringRef getDescription() const override {
    return "Insert CC->workspace(GM)->UB bridge + set_ffts + signature tweak (guarded)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.getName() != "cube_matmul_vadd_2d")
      return;

    MatchInfo mi;
    if (!match(func, mi))
      return;

    IRRewriter rewriter(func.getContext());
    Location loc = func.getLoc();
    Block &entry = func.front();

    llvm::errs() << "\n[InsertLoadStoreForMixCV] MATCH func: " << func.getName()
                 << "\n---- BEFORE ----\n";
    func.print(llvm::errs());
    llvm::errs() << "\n--------------\n";

    // (A) force tile_idx_x/y -> 0 and erase them (original args[9],[10])
    Value c0 = getOrCreateC0(rewriter, func).getResult();
    {
      Value tileX = entry.getArgument(9);
      Value tileY = entry.getArgument(10);
      tileX.replaceAllUsesWith(c0);
      tileY.replaceAllUsesWith(c0);
      func.eraseArgument(10);
      func.eraseArgument(9);
    }

    // (B) inherit gmSpace from existing GM subview type
    auto gmSubviewTy = dyn_cast<MemRefType>(mi.cGMSubview.getType());
    if (!gmSubviewTy || !gmSubviewTy.getMemorySpace()) {
      llvm::errs() << "[InsertLoadStoreForMixCV] ERROR: cannot get gm memorySpace\n";
      signalPassFailure();
      return;
    }
    Attribute gmSpace = gmSubviewTy.getMemorySpace();

    // (C) create ubSpace using enum API
    Attribute ubSpace = pto::AddressSpaceAttr::get(func.getContext(), pto::AddressSpace::VEC);
    if (!ubSpace) {
      llvm::errs() << "[InsertLoadStoreForMixCV] ERROR: cannot create ub address_space attr\n";
      signalPassFailure();
      return;
    }

    // (D) insert new args (ws, ffts) at front
    MemRefType wsArgTy =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{}, gmSpace);
    MemRefType fftsArgTy =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type(),
                        MemRefLayoutAttrInterface{}, gmSpace);

    func.insertArgument(0, wsArgTy, DictionaryAttr{}, loc);
    func.insertArgument(1, fftsArgTy, DictionaryAttr{}, loc);

    Value argWS = entry.getArgument(0);
    Value argFFTs = entry.getArgument(1);

    // (E) insert pto.set_ffts near top (after constants)
    {
      Operation *insertPt = &entry.front();
      for (Operation &op : entry.getOperations()) {
        if (isa<arith::ConstantOp>(op) || isa<arith::ConstantIndexOp>(op)) {
          insertPt = op.getNextNode() ? op.getNextNode() : &op;
          continue;
        }
        break;
      }
      rewriter.setInsertionPoint(insertPt);
      rewriter.create<pto::SetFFTsOp>(loc, argFFTs);
    }

    // (F) types
    MemRefType wsTileTy =
        MemRefType::get({32, 32}, rewriter.getF32Type(),
                        MemRefLayoutAttrInterface{}, gmSpace);
    MemRefType ubTileTy =
        MemRefType::get({32, 32}, rewriter.getF32Type(),
                        MemRefLayoutAttrInterface{}, ubSpace);

    // (G) insert bridge after matmul
    rewriter.setInsertionPointAfter(mi.matmulDps);

    // workspace view: memref.view %arg_ws -> memref<32x32xf32, gm>
    // byte_shift = 0; sizes empty (no dynamic dims).
    Value wsTile = rewriter.create<memref::ViewOp>(
        loc, wsTileTy, argWS, c0, ValueRange{}).getResult();

    // CC -> workspace(GM)
    rewriter.create<pto::TStoreOp>(loc, TypeRange{}, mi.mmCCBuf, wsTile);

    // workspace(GM) -> UB
    auto tmatUb = createAllocWithAlign(rewriter, loc, ubTileTy, 64);
    rewriter.create<pto::TLoadOp>(loc, TypeRange{}, wsTile, tmatUb.getResult());

    // C(GM) -> UB
    auto cUb = createAllocWithAlign(rewriter, loc, ubTileTy, 64);
    rewriter.create<pto::TLoadOp>(loc, TypeRange{}, mi.cGMSubview, cUb.getResult());

    // addf on UB
    auto outUb = createAllocWithAlign(rewriter, loc, ubTileTy, 64);
    rewriter.create<pto::AddFDpsOp>(loc, TypeRange{}, cUb.getResult(),
                                    tmatUb.getResult(), outUb.getResult());

    // UB -> OUT(GM)
    rewriter.create<pto::TStoreOp>(loc, TypeRange{}, outUb.getResult(), mi.outGMSubview);

    // (H) erase old tail ops
    rewriter.eraseOp(mi.storeToOut);
    rewriter.eraseOp(mi.addfCbuf);
    rewriter.eraseOp(mi.ccToVec);
    rewriter.eraseOp(mi.cGmToCbuf);

    llvm::errs() << "---- AFTER ----\n";
    func.print(llvm::errs());
    llvm::errs() << "\n==============\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createPTOInsertLoadStoreForMixCVPass() {
  return std::make_unique<InsertLoadStoreForMixCVPass>();
}

} // namespace pto
} // namespace mlir
