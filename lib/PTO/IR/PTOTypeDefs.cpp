//===- PTOTypeDefs.cpp --------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::pto;

TileBufConfigAttr TileBufType::getConfigAttr() const {
  // 情况 A：getConfig() 已经是 TileBufConfigAttr
  if constexpr (std::is_same_v<decltype(getConfig()), TileBufConfigAttr>) {
    auto cfg = getConfig();
    if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());
    return cfg;
  } else {
    // 情况 B：getConfig() 是 Attribute
    auto cfg = llvm::dyn_cast_or_null<TileBufConfigAttr>(getConfig());
    if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());
    return cfg;
  }
}
bool TileBufType::hasNonDefaultConfig() const {
  return !getConfigAttr().isDefault();
}

mlir::Attribute TileBufType::getBLayoutAttr() const { return getConfigAttr().getBLayout(); }
mlir::Attribute TileBufType::getSLayoutAttr() const { return getConfigAttr().getSLayout(); }
mlir::Attribute TileBufType::getPadValueAttr() const { return getConfigAttr().getPad(); }

// ✅ numeric getters（可选）
int32_t TileBufType::getSFractalSizeI32() const {
  return (int32_t)getConfigAttr().getSFractalSize().getInt();
}

int32_t TileBufType::getBLayoutValueI32() const {
  if (auto a = llvm::dyn_cast<BLayoutAttr>(getBLayoutAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getSLayoutValueI32() const {
  if (auto a = llvm::dyn_cast<SLayoutAttr>(getSLayoutAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getPadValueI32() const {
  if (auto a = llvm::dyn_cast<PadValueAttr>(getPadValueAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

// ---- TileBufType custom asm ----
// !pto.tile_buf<<loc=.., dtype=.., rows=.., cols=.., blayout=.., valid=..x.., slayout=.., fractal=.., pad=..>>
Type TileBufType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  if (failed(parser.parseLess()))
    return Type();

  std::string locStr;
  Type dtype;
  int64_t rows = 0, cols = 0;
  int64_t vrow = -1, vcol = -1;
  std::string blayoutStr, slayoutStr;
  int64_t fractal = 0;
  uint32_t padInt;

  auto parseKeyEq = [&](StringRef expectedKey) -> LogicalResult {
    if (failed(parser.parseKeyword(expectedKey)))
      return failure();
    if (failed(parser.parseEqual()))
      return failure();
    return success();
  };

  // loc=Vec
  {
    if (failed(parseKeyEq("loc"))) return Type();
    // Vec/Mat/Acc 不是类型/属性，直接当 keyword/string 读
    if (failed(parser.parseKeywordOrString(&locStr))) return Type();
    if (failed(parser.parseComma())) return Type();
  }

  // dtype=f16
  {
    if (failed(parseKeyEq("dtype"))) return Type();
    if (failed(parser.parseType(dtype))) return Type();
    if (failed(parser.parseComma())) return Type();
  }

  // rows=16
  {
    if (failed(parseKeyEq("rows"))) return Type();
    if (failed(parser.parseInteger(rows))) return Type();
    if (failed(parser.parseComma())) return Type();
  }

  // cols=16
  {
    if (failed(parseKeyEq("cols"))) return Type();
    if (failed(parser.parseInteger(cols))) return Type();
    if (failed(parser.parseComma())) return Type();
  }
   
  {
    // v_row=?/-1/16 , v_col=?/-1/8   （支持半动态）
    if (failed(parseKeyEq("v_row"))) return Type();

    // 解析 v_row：'?' -> -1，否则整数（允许 -1 兼容）
    if (succeeded(parser.parseOptionalQuestion())) {
        vrow = -1;
    } else {
        if (failed(parser.parseInteger(vrow))) return Type();
        if (vrow < -1) {
            parser.emitError(parser.getCurrentLocation(),
                            "v_row must be '?', -1, or a non-negative integer");
            return Type();
        }
    }

    if (failed(parser.parseComma())) return Type();

    if (failed(parseKeyEq("v_col"))) return Type();

    // 解析 v_col：'?' -> -1，否则整数（允许 -1 兼容）
    if (succeeded(parser.parseOptionalQuestion())) {
        vcol = -1;
    } else {
        if (failed(parser.parseInteger(vcol))) return Type();
        if (vcol < -1) {
            parser.emitError(parser.getCurrentLocation(),
                            "v_col must be '?', -1, or a non-negative integer");
            return Type();
        }
    }
    if (failed(parser.parseComma())) return Type();
  }

  // blayout=RowMajor
  {
    if (failed(parseKeyEq("blayout"))) return Type();
    if (failed(parser.parseKeywordOrString(&blayoutStr))) return Type();
    if (failed(parser.parseComma())) return Type();
  }


  // slayout=NoneBox
  {
    if (failed(parseKeyEq("slayout"))) return Type();
    if (failed(parser.parseKeywordOrString(&slayoutStr))) return Type();
    if (failed(parser.parseComma())) return Type();
  }

  // fractal=512
  {
    if (failed(parseKeyEq("fractal"))) return Type();
    if (failed(parser.parseInteger(fractal))) return Type();
    if (failed(parser.parseComma())) return Type();
  }

  // pad=Null
  {
    if (failed(parseKeyEq("pad"))) return Type();
    if (failed(parser.parseInteger(padInt))) return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  // -------- 语义校验/构造 --------
  if (rows < 0 || cols < 0) {
    parser.emitError(parser.getNameLoc(), "rows/cols must be non-negative");
    return Type();
  }

  auto memorySpace = ::llvm::StringSwitch<::std::optional<AddressSpace>>(locStr)
        .Case("mat", AddressSpace::MAT)
        .Case("left", AddressSpace::LEFT)
        .Case("right", AddressSpace::RIGHT)
        .Case("acc", AddressSpace::ACC)
        .Case("vec", AddressSpace::VEC)
        .Case("bias", AddressSpace::BIAS)
        .Case("scaling", AddressSpace::SCALING)
        .Default(::std::nullopt);
  if (!memorySpace.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown loc: ") << locStr;
    return Type();
  }

  auto bl = symbolizeBLayout(blayoutStr);
  auto sl = symbolizeSLayout(slayoutStr);
  auto pv = symbolizePadValue(padInt);
  if (!bl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown blayout: ") << blayoutStr;
    return Type();
  }
  if (!sl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown slayout: ") << slayoutStr;
    return Type();
  }
  if (!pv.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown pad: ") << padInt;
    return Type();
  }

  auto blAttr = BLayoutAttr::get(ctx, bl.value());
  auto slAttr = SLayoutAttr::get(ctx, sl.value());
  auto fractalAttr =
      IntegerAttr::get(IntegerType::get(ctx, 32), fractal);
  auto padAttr = PadValueAttr::get(ctx, pv.value());
  auto memorySpaceAttr = AddressSpaceAttr::get(ctx, memorySpace.value());
  auto cfg = TileBufConfigAttr::get(ctx, blAttr, slAttr, fractalAttr, padAttr);

  SmallVector<int64_t, 2> shape{rows, cols};
  SmallVector<int64_t, 2> validShape{vrow, vcol};

  return TileBufType::get(ctx, shape, dtype, memorySpaceAttr, llvm::ArrayRef<int64_t>(validShape), cfg);
}

static llvm::StringRef stringifyLocFromMemorySpace(mlir::Attribute memorySpace) {
  auto asAttr = llvm::dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  switch (asAttr.getAddressSpace()) {
    case AddressSpace::MAT: return "mat";
    case AddressSpace::LEFT: return "left";
    case AddressSpace::RIGHT: return "right";
    case AddressSpace::ACC: return "acc";
    case AddressSpace::VEC: return "vec";
    case AddressSpace::BIAS: return "bias";
    case AddressSpace::SCALING: return "scaling";
    default: return "illegal";
  }
}

static llvm::StringRef stringifyLocFromPad(mlir::Attribute pad) {
  auto padAttr = llvm::dyn_cast_or_null<PadValueAttr>(pad);
  if (!padAttr) return "9999";

  switch (padAttr.getValue()) {
    case PadValue::Null: return "0";
    case PadValue::Zero: return "1";
    case PadValue::Max: return "2";
    case PadValue::Min: return "3";
    default:
      return "9999";
  }
}

void mlir::pto::TileBufType::print(mlir::AsmPrinter &printer) const {
    auto shape = getShape();
    int64_t rows = shape.size() > 0 ? shape[0] : 0;
    int64_t cols = shape.size() > 1 ? shape[1] : 0;

    auto cfg = getConfigAttr();
    if (!cfg) cfg = mlir::pto::TileBufConfigAttr::getDefault(getContext());

    llvm::StringRef locStr = stringifyLocFromMemorySpace(getMemorySpace());

    printer << "<"
            << "loc=" << locStr
            << ", dtype=";
    printer.printType(getElementType());

    auto blayout = llvm::dyn_cast<BLayoutAttr>(cfg.getBLayout());
    auto slayout = llvm::dyn_cast<SLayoutAttr>(cfg.getSLayout());

    auto vs = getValidShape(); // ArrayRef<int64_t>
    int64_t vrow = rows;
    int64_t vcol = cols;

    if (vs.size() >= 2) {
        vrow = vs[0];
        vcol = vs[1];
    }
    printer << ", rows=" << rows
            << ", cols=" << cols;
    printer << ", v_row=";
    if (vrow < 0) printer << "?";
    else printer << vrow;

    printer << ", v_col=";
    if (vcol < 0) printer << "?";
    else printer << vcol;

    printer << ", blayout=" << stringifyBLayout(blayout.getValue())
        << ", slayout=" << stringifySLayout(slayout.getValue())
        << ", fractal=" << cfg.getSFractalSize().getInt()
        << ", pad=" << stringifyLocFromPad(cfg.getPad())
        << ">";
}