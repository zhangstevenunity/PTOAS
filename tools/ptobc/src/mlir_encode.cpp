#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_format.h"

#include "ptobc/leb128.h"
#include "ptobc_opcodes_v0.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <PTO/IR/PTO.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/SourceMgr.h>

#include <llvm/ADT/DenseMap.h>

#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace ptobc {

static uint64_t internType(PTOBCFile& f, mlir::Type t) {
  std::string s = printType(t);
  f.strings.intern(s);
  // type ids are 1-based
  for (size_t i = 0; i < f.typeAsm.size(); ++i) {
    if (f.typeAsm[i] == s) return i + 1;
  }
  f.typeAsm.push_back(s);
  return f.typeAsm.size();
}

static mlir::DictionaryAttr stripAttr(mlir::MLIRContext *ctx, mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (!dict) return dict;
  if (!dict.get(key)) return dict;
  llvm::SmallVector<mlir::NamedAttribute, 8> keep;
  keep.reserve(dict.size());
  for (auto na : dict) {
    if (na.getName().getValue() == key) continue;
    keep.push_back(na);
  }
  return mlir::DictionaryAttr::get(ctx, keep);
}

static uint64_t internAttr(PTOBCFile& f, mlir::DictionaryAttr dict) {
  if (!dict || dict.empty()) return 0;
  std::string s = printAttrDict(dict);
  f.strings.intern(s);
  for (size_t i = 0; i < f.attrAsm.size(); ++i) {
    if (f.attrAsm[i] == s) return i + 1;
  }
  f.attrAsm.push_back(s);
  return f.attrAsm.size();
}

static std::string hexFloatLiteral(mlir::FloatAttr a) {
  llvm::SmallVector<char, 32> digits;
  llvm::APInt bits = a.getValue().bitcastToAPInt();
  bits.toString(digits, /*Radix=*/16, /*Signed=*/false, /*formatAsCLiteral=*/true);
  return std::string(digits.data(), digits.size());
}

static std::string apIntToSignedDecimal(const llvm::APInt &v) {
  llvm::SmallVector<char, 32> digits;
  v.toString(digits, /*Radix=*/10, /*Signed=*/true, /*formatAsCLiteral=*/false);
  return std::string(digits.data(), digits.size());
}

struct Encoder {
  PTOBCFile file;

  bool emitDebugInfo = false;
  bool allowGeneric = false;

  // constpool dedup: key is raw bytes: tag + payload
  std::unordered_map<std::string, uint64_t> constIdByKey;

  // Per-function numbering state.
  uint64_t funcId = 0;
  uint64_t nextOpId = 0;
  llvm::DenseMap<mlir::Value, uint64_t> valueId;
  std::vector<mlir::Value> valueById;

  // Module-wide debug file table state.
  std::unordered_map<std::string, uint64_t> dbgFileIdByPath;

  uint64_t getValueId(mlir::Value v) {
    auto it = valueId.find(v);
    if (it == valueId.end()) {
      throw std::runtime_error("operand references undefined value");
    }
    return it->second;
  }

  uint64_t allocValueId(mlir::Value v) {
    uint64_t id = valueId.size();
    auto [it, inserted] = valueId.try_emplace(v, id);
    if (!inserted) throw std::runtime_error("value already has an id");
    valueById.push_back(v);
    return it->second;
  }

  uint64_t internDbgFile(llvm::StringRef path) {
    auto p = path.str();
    auto it = dbgFileIdByPath.find(p);
    if (it != dbgFileIdByPath.end()) return it->second;

    uint64_t sid = file.strings.intern(p);
    uint64_t fileId = file.dbgFiles.size();
    file.dbgFiles.push_back(DebugFileEntry{sid, /*hashKind=*/0, {}});
    dbgFileIdByPath.emplace(std::move(p), fileId);
    return fileId;
  }

  void recordOpLocation(uint64_t opId, mlir::Operation &op) {
    if (!emitDebugInfo) return;
    auto loc = op.getLoc();
    auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(loc);
    if (!flc) return;

    uint64_t fileId = internDbgFile(flc.getFilename().getValue());
    uint64_t sl = flc.getLine();
    uint64_t sc = flc.getColumn();
    uint64_t el = sl;
    uint64_t ec = sc + 1; // point-range

    file.dbgLocations.push_back(DebugLocationEntry{funcId, opId, fileId, sl, sc, el, ec});
  }

  void finalizeValueNamesForFunction() {
    if (!emitDebugInfo) return;
    // Deterministic value names for DebugInfo.
    std::unordered_map<std::string, int> constCounts;

    for (uint64_t vid = 0; vid < valueById.size(); ++vid) {
      mlir::Value v = valueById[vid];
      std::string name;

      if (auto *def = v.getDefiningOp()) {
        if (auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(def)) {
          mlir::Attribute a = cst.getValue();
          std::string ty = printType(v.getType());

          // Only generate special names for scalar ints/floats.
          if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(a)) {
            std::string imm = hexFloatLiteral(fa);
            std::string base = "c" + imm + "_" + ty;
            int &n = constCounts[base];
            name = base;
            if (n > 0) name += "_" + std::to_string(n);
            ++n;
          } else if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(a)) {
            std::string imm = apIntToSignedDecimal(ia.getValue());
            std::string base = "c" + imm;
            if (ty != "index") base += "_" + ty;
            int &n = constCounts[base];
            name = base;
            if (n > 0) name += "_" + std::to_string(n);
            ++n;
          }
        }
      }

      if (name.empty()) {
        // Non-constant (or non-scalar-constant) value.
        name = std::to_string(vid);
      }

      uint64_t nameSid = file.strings.intern(name);
      file.dbgValueNames.push_back(DebugValueNameEntry{funcId, vid, nameSid});
    }
  }

  uint64_t internConst(uint8_t tag, const std::vector<uint8_t> &payload) {
    std::string key;
    key.resize(1 + payload.size());
    key[0] = char(tag);
    if (!payload.empty()) {
      std::memcpy(key.data() + 1, payload.data(), payload.size());
    }
    auto it = constIdByKey.find(key);
    if (it != constIdByKey.end()) return it->second;
    uint64_t id = file.consts.size();
    file.consts.push_back(ConstEntry{tag, payload});
    constIdByKey.emplace(std::move(key), id);
    return id;
  }

  uint64_t internConstInt64(uint64_t typeId, int64_t value) {
    Buffer p;
    writeULEB128(typeId, p.bytes);
    writeSLEB128(value, p.bytes);
    return internConst(/*tag=*/0x01, p.bytes);
  }

  uint64_t internConstIntBits(uint64_t typeId, const llvm::APInt &bits) {
    Buffer p;
    writeULEB128(typeId, p.bytes);

    const unsigned byteLen = (bits.getBitWidth() + 7) / 8;
    writeULEB128(byteLen, p.bytes);

    // little-endian bytes
    llvm::SmallVector<uint64_t, 4> words;
    words.resize(bits.getNumWords());
    std::memcpy(words.data(), bits.getRawData(), words.size() * sizeof(uint64_t));

    for (unsigned i = 0; i < byteLen; ++i) {
      unsigned word = i / 8;
      unsigned off = (i % 8) * 8;
      uint8_t b = uint8_t((words[word] >> off) & 0xFFu);
      p.bytes.push_back(b);
    }

    return internConst(/*tag=*/0x04, p.bytes);
  }

  uint64_t internConstFloatBits(uint64_t dtypeId, const llvm::APInt &bits) {
    Buffer p;
    writeULEB128(dtypeId, p.bytes);
    const unsigned byteLen = (bits.getBitWidth() + 7) / 8;
    writeULEB128(byteLen, p.bytes);

    // little-endian bytes
    llvm::SmallVector<uint64_t, 4> words;
    words.resize(bits.getNumWords());
    std::memcpy(words.data(), bits.getRawData(), words.size() * sizeof(uint64_t));

    for (unsigned i = 0; i < byteLen; ++i) {
      unsigned word = i / 8;
      unsigned off = (i % 8) * 8;
      uint8_t b = uint8_t((words[word] >> off) & 0xFFu);
      p.bytes.push_back(b);
    }

    return internConst(/*tag=*/0x02, p.bytes);
  }

  void resetForFunction(uint64_t fid) {
    funcId = fid;
    nextOpId = 0;
    valueId.clear();
    valueById.clear();
  }

  void encodeRegion(mlir::Region& region, Buffer& out);
  void encodeBlock(mlir::Block& block, Buffer& out);
  void encodeOp(mlir::Operation& op, Buffer& out);
};

void Encoder::encodeRegion(mlir::Region& region, Buffer& out) {
  writeULEB128(region.getBlocks().size(), out.bytes);
  for (auto& block : region.getBlocks()) {
    encodeBlock(block, out);
  }
}

void Encoder::encodeBlock(mlir::Block& block, Buffer& out) {
  // block args
  writeULEB128(block.getNumArguments(), out.bytes);
  for (auto arg : block.getArguments()) {
    writeULEB128(internType(file, arg.getType()), out.bytes);
    allocValueId(arg);
  }

  // ops count
  size_t opCount = 0;
  for (auto& op : block.getOperations()) (void)op, ++opCount;
  writeULEB128(opCount, out.bytes);

  for (auto& op : block.getOperations()) {
    encodeOp(op, out);
  }
}

void Encoder::encodeOp(mlir::Operation& op, Buffer& out) {
  if (emitDebugInfo) {
    // op_id (preorder DFS, per-function)
    uint64_t opId = nextOpId++;
    recordOpLocation(opId, op);
  }

  // Try compact known-op encoding first (PTO-BC v0).
  auto fullName = op.getName().getStringRef();
  auto ov = ptobc::v0::lookupOpcodeAndVariantByFullName(fullName);
  if (ov) {
    const auto *info = ptobc::v0::lookupByOpcode(ov->opcode);
    if (!info) throw std::runtime_error("missing v0 opcode schema for op: " + fullName.str());

    // Allocate value IDs for results first so nested regions can reference them.
    const uint64_t resStart = valueId.size();
    for (auto res : op.getResults()) {
      allocValueId(res);
    }

    // u16 opcode
    out.appendU16LE(ov->opcode);

    // attr_id (allow per-op stripping)
    mlir::DictionaryAttr dict = op.getAttrDictionary();

    // arith.constant: value is encoded via CONSTPOOL (imm_kind=0x05)
    if (auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(&op)) {
      dict = stripAttr(op.getContext(), dict, "value");
    }

    auto attrId = internAttr(file, dict);
    writeULEB128(attrId, out.bytes);

    // variant u8
    if (info->has_variant_u8) {
      out.appendU8(ov->variant);
    }

    // immediates
    llvm::SmallVector<uint64_t, 4> imms;
    imms.clear();

    if (info->imm_kind == 0x00) {
      // none
    } else if (info->imm_kind == 0x01) {
      // arith.cmpi predicate
      auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(&op);
      if (!cmp) throw std::runtime_error("imm_kind=cmpi_pred but op is not arith.cmpi");
      uint8_t p;
      switch (cmp.getPredicate()) {
        case mlir::arith::CmpIPredicate::eq: p = 0; break;
        case mlir::arith::CmpIPredicate::ne: p = 1; break;
        case mlir::arith::CmpIPredicate::slt: p = 2; break;
        case mlir::arith::CmpIPredicate::sle: p = 3; break;
        case mlir::arith::CmpIPredicate::sgt: p = 4; break;
        case mlir::arith::CmpIPredicate::sge: p = 5; break;
        default:
          throw std::runtime_error("unsupported arith.cmpi predicate (v0 supports only eq/ne/slt/sle/sgt/sge)");
      }
      out.appendU8(p);
      imms.push_back(p);
    } else if (info->imm_kind == 0x02) {
      // record_event/wait_event: event3(u8,u8,u8)
      auto src = op.getAttrOfType<mlir::pto::SyncOpTypeAttr>("src_op");
      auto dst = op.getAttrOfType<mlir::pto::SyncOpTypeAttr>("dst_op");
      auto eid = op.getAttrOfType<mlir::pto::EventAttr>("event_id");
      if (!src || !dst || !eid) throw std::runtime_error("event op missing src_op/dst_op/event_id attrs");
      uint8_t a = uint8_t(src.getOpType());
      uint8_t b = uint8_t(dst.getOpType());
      uint8_t c = uint8_t(eid.getEvent());
      out.appendU8(a);
      out.appendU8(b);
      out.appendU8(c);
      imms.push_back(a);
      imms.push_back(b);
      imms.push_back(c);
    } else if (info->imm_kind == 0x05) {
      // arith.constant: const_id(uLEB128)
      auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(&op);
      if (!cst) throw std::runtime_error("imm_kind=const_id but op is not arith.constant");

      mlir::Attribute a = cst.getValue();
      uint64_t cid = 0;
      if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(a)) {
        uint64_t typeId = internType(file, cst.getType());
        const llvm::APInt &v = ia.getValue();
        if (v.getBitWidth() <= 64) {
          cid = internConstInt64(typeId, v.getSExtValue());
        } else {
          cid = internConstIntBits(typeId, v);
        }
      } else if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(a)) {
        uint64_t dtypeId = internType(file, cst.getType());
        cid = internConstFloatBits(dtypeId, fa.getValue().bitcastToAPInt());
      } else {
        throw std::runtime_error("unsupported arith.constant attribute kind for compact v0");
      }
      writeULEB128(cid, out.bytes);
      imms.push_back(cid);
    } else if (info->imm_kind == 0x06) {
      // make_tensor_view: list_mode(u8), nshape(uLEB), nstrides(uLEB)
      auto mtv = llvm::dyn_cast<mlir::pto::MakeTensorViewOp>(&op);
      if (!mtv) throw std::runtime_error("imm_kind=make_tensor_view but op is not pto.make_tensor_view");
      uint8_t lm = 0; // list_mode=0 (inline value_ids)
      out.appendU8(lm);
      writeULEB128(mtv.getShape().size(), out.bytes);
      writeULEB128(mtv.getStrides().size(), out.bytes);
      imms.push_back(lm);
      imms.push_back(mtv.getShape().size());
      imms.push_back(mtv.getStrides().size());
    } else if (info->imm_kind == 0x07) {
      // partition_view: list_mode(u8), noffsets(uLEB), nsizes(uLEB)
      auto pv = llvm::dyn_cast<mlir::pto::PartitionViewOp>(&op);
      if (!pv) throw std::runtime_error("imm_kind=partition_view but op is not pto.partition_view");
      uint8_t lm = 0;
      out.appendU8(lm);
      writeULEB128(pv.getOffsets().size(), out.bytes);
      writeULEB128(pv.getSizes().size(), out.bytes);
      imms.push_back(lm);
      imms.push_back(pv.getOffsets().size());
      imms.push_back(pv.getSizes().size());
    } else if (info->imm_kind == 0x08) {
      // alloc_tile: optmask(u8)
      auto at = llvm::dyn_cast<mlir::pto::AllocTileOp>(&op);
      if (!at) throw std::runtime_error("imm_kind=alloc_tile but op is not pto.alloc_tile");
      uint8_t mask = 0;
      if (at.getValidRow()) mask |= 0x1;
      if (at.getValidCol()) mask |= 0x2;
      out.appendU8(mask);
      imms.push_back(mask);
    } else {
      throw std::runtime_error("unknown imm_kind in v0 schema");
    }

    // operands
    auto emitOperands = [&](size_t n) {
      if (op.getNumOperands() != n) {
        throw std::runtime_error("operand count mismatch for op: " + fullName.str());
      }
      for (auto v : op.getOperands()) {
        writeULEB128(getValueId(v), out.bytes);
      }
    };

    if (info->operand_mode == 0x00) {
      emitOperands(info->num_operands);

    } else if (info->operand_mode == 0x01) {
      auto n = ptobc::v0::lookupOperandsByVariant(ov->opcode, ov->variant);
      if (!n) throw std::runtime_error("missing by-variant operand count");
      emitOperands(*n);

    } else if (info->operand_mode == 0x02) {
      writeULEB128(op.getNumOperands(), out.bytes);
      for (auto v : op.getOperands()) {
        writeULEB128(getValueId(v), out.bytes);
      }

    } else if (info->operand_mode == 0x03) {
      // segmented (inline list_mode=0 only)
      if (imms.size() < 3) throw std::runtime_error("segmented operands missing immediates");
      if (imms[0] != 0) throw std::runtime_error("list_mode=1 not implemented in ptobc encoder yet");
      const size_t base = info->num_operands;
      const size_t n1 = size_t(imms[1]);
      const size_t n2 = size_t(imms[2]);
      emitOperands(base + n1 + n2);

    } else if (info->operand_mode == 0x04) {
      // optional mask2
      if (imms.empty()) throw std::runtime_error("optmask operands missing immediate");
      uint8_t mask = uint8_t(imms[0]);
      size_t n = ((mask & 0x1) ? 1 : 0) + ((mask & 0x2) ? 1 : 0);
      emitOperands(n);

    } else {
      throw std::runtime_error("unknown operand_mode in v0 schema");
    }

    // explicit result type ids
    if (info->result_type_mode == 0x01) {
      if (op.getNumResults() != info->num_results) {
        throw std::runtime_error("result count mismatch for op: " + fullName.str());
      }
      for (auto res : op.getResults()) {
        writeULEB128(internType(file, res.getType()), out.bytes);
      }
    }

    // regions
    if (op.getNumRegions() != info->num_regions) {
      throw std::runtime_error("region count mismatch for op: " + fullName.str());
    }
    for (auto &r : op.getRegions()) {
      encodeRegion(r, out);
    }

    (void)resStart;
    return;
  }

  if (!allowGeneric) {
    throw std::runtime_error("op is not in v0 opcode table (and PTOBC_ALLOW_GENERIC is not set): " + fullName.str());
  }

  // === Generic op escape ===
  out.appendU16LE(kOpcodeGeneric);

  // attr_id
  auto attrId = internAttr(file, op.getAttrDictionary());
  writeULEB128(attrId, out.bytes);

  // op-name
  auto opName = op.getName().getStringRef().str();
  auto opNameSid = file.strings.intern(opName);
  writeULEB128(opNameSid, out.bytes);

  // results
  writeULEB128(op.getNumResults(), out.bytes);
  for (auto res : op.getResults()) {
    allocValueId(res);
    writeULEB128(internType(file, res.getType()), out.bytes);
  }

  // operands
  writeULEB128(op.getNumOperands(), out.bytes);
  for (auto operand : op.getOperands()) {
    writeULEB128(getValueId(operand), out.bytes);
  }

  // regions
  writeULEB128(op.getNumRegions(), out.bytes);
  for (auto& r : op.getRegions()) {
    encodeRegion(r, out);
  }
}

PTOBCFile encodeFromMLIRModule(mlir::ModuleOp module) {
  Encoder enc;
  enc.emitDebugInfo = (std::getenv("PTOBC_EMIT_DEBUGINFO") != nullptr);
  enc.allowGeneric = (std::getenv("PTOBC_ALLOW_GENERIC") != nullptr);

  // Pre-intern a few common strings to stabilize ids.
  enc.file.strings.intern("func.func");
  enc.file.strings.intern("func.return");

  // MODULE encoding
  Buffer m;
  // profile_id=0 (unspecified), index_width=64
  m.appendU8(0);
  m.appendU8(64);

  // module_attr_id
  uint64_t modAttrId = internAttr(enc.file, module->getAttrDictionary());
  writeULEB128(modAttrId, m.bytes);

  // globals count
  writeULEB128(0, m.bytes);

  // function decls (top-level order)
  llvm::SmallVector<mlir::func::FuncOp, 8> funcs;
  for (auto f : module.getOps<mlir::func::FuncOp>()) {
    funcs.push_back(f);
  }

  writeULEB128(funcs.size(), m.bytes);

  // encode decls
  for (auto f : funcs) {
    auto nameSid = enc.file.strings.intern(f.getName().str());
    // func type as opaque asm in type table
    auto funcTypeId = internType(enc.file, f.getFunctionType());
    // flags: bit0 import? (0)
    uint8_t flags = 0;
    auto funcAttrId = internAttr(enc.file, f->getAttrDictionary());

    writeULEB128(nameSid, m.bytes);
    writeULEB128(funcTypeId, m.bytes);
    m.appendU8(flags);
    writeULEB128(funcAttrId, m.bytes);
  }

  // bodies: for each function, encode its body region
  for (size_t i = 0; i < funcs.size(); ++i) {
    auto f = funcs[i];
    enc.resetForFunction(i);

    // function body is region #0
    enc.encodeRegion(f.getBody(), m);

    // DebugInfo: deterministic value names for this function.
    enc.finalizeValueNamesForFunction();
  }

  enc.file.moduleBytes = std::move(m.bytes);
  return enc.file;
}

mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path) {
  llvm::SourceMgr sm;
  std::string err;
  auto file = mlir::openInputFile(path, &err);
  if (!file) {
    throw std::runtime_error("failed to open input: " + path + (err.empty() ? "" : (": " + err)));
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &ctx);
  if (!module) {
    throw std::runtime_error("failed to parse MLIR file: " + path);
  }
  return module;
}

} // namespace ptobc
