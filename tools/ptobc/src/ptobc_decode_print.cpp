#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_format.h"
#include "ptobc/leb128.h"
#include "ptobc/canonical_printer.h"
#include "ptobc/ptobc_decode.h"
#include "ptobc_opcodes_v0.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <PTO/IR/PTO.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

namespace ptobc {

static bool debugEnabled() {
  return std::getenv("PTOBC_DEBUG") != nullptr;
}

struct Reader {
  const uint8_t* p;
  const uint8_t* end;

  uint8_t readU8() {
    if (p >= end) throw std::runtime_error("EOF");
    return *p++;
  }
  uint16_t readU16LE() {
    uint16_t lo = readU8();
    uint16_t hi = readU8();
    return lo | (hi << 8);
  }
  uint32_t readU32LE() {
    uint32_t b0 = readU8();
    uint32_t b1 = readU8();
    uint32_t b2 = readU8();
    uint32_t b3 = readU8();
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }
  uint64_t readULEB() {
    uint64_t v;
    size_t n = readULEB128(p, size_t(end - p), v);
    p += n;
    return v;
  }

  int64_t readSLEB() {
    int64_t v;
    size_t n = readSLEB128(p, size_t(end - p), v);
    p += n;
    return v;
  }

  std::vector<uint8_t> readBytes(size_t n) {
    if (size_t(end - p) < n) throw std::runtime_error("EOF");
    std::vector<uint8_t> out(p, p + n);
    p += n;
    return out;
  }
};

static void parseStringsSection(const std::vector<uint8_t>& data, std::vector<std::string>& strings) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  strings.clear();
  strings.reserve(cnt);
  for (uint64_t i = 0; i < cnt; ++i) {
    uint64_t len = r.readULEB();
    auto bs = r.readBytes(len);
    strings.emplace_back(reinterpret_cast<const char*>(bs.data()), bs.size());
  }
  if (r.p != r.end) throw std::runtime_error("trailing bytes in STRINGS");
}

struct TypeEntry { uint8_t tag; std::string asmStr; };
struct AttrEntry { uint8_t tag; std::string asmStr; };

struct ConstEntryParsed {
  uint8_t tag;
  // tag=0x01 int: type_id + sLEB128
  uint64_t typeId = 0;
  int64_t intValue = 0;
  // tag=0x02 float bits: type_id + bytes
  std::vector<uint8_t> floatBytes;
  // tag=0x04 wide int bits: type_id + bytes
  std::vector<uint8_t> intBytes;
};

struct DbgFileEntry { uint64_t pathSid; uint8_t hashKind; std::vector<uint8_t> hashBytes; };
struct DbgValueNameEntry { uint64_t funcId; uint64_t valueId; uint64_t nameSid; };
struct DbgLocationEntry { uint64_t funcId; uint64_t opId; uint64_t fileId; uint64_t sl; uint64_t sc; uint64_t el; uint64_t ec; };
struct DbgSnippetEntry { uint64_t funcId; uint64_t opId; uint64_t snippetSid; };

struct DebugInfo {
  std::vector<DbgFileEntry> files;
  std::vector<DbgValueNameEntry> valueNames;
  std::vector<DbgLocationEntry> locations;
  std::vector<DbgSnippetEntry> snippets;
};

static DebugInfo parseDebugInfoSection(const std::vector<uint8_t>& data) {
  Reader r{data.data(), data.data() + data.size()};
  DebugInfo di;

  // FileTable
  uint64_t fcnt = r.readULEB();
  di.files.reserve(fcnt);
  for (uint64_t i = 0; i < fcnt; ++i) {
    uint64_t psid = r.readULEB();
    uint8_t hk = r.readU8();
    std::vector<uint8_t> hb;
    if (hk != 0) {
      uint64_t hlen = r.readULEB();
      hb = r.readBytes(hlen);
    }
    di.files.push_back({psid, hk, std::move(hb)});
  }

  // ValueNames
  uint64_t vcnt = r.readULEB();
  di.valueNames.reserve(vcnt);
  for (uint64_t i = 0; i < vcnt; ++i) {
    uint64_t fid = r.readULEB();
    uint64_t vid = r.readULEB();
    uint64_t nsid = r.readULEB();
    di.valueNames.push_back({fid, vid, nsid});
  }

  // OpLocations
  uint64_t lcnt = r.readULEB();
  di.locations.reserve(lcnt);
  for (uint64_t i = 0; i < lcnt; ++i) {
    uint64_t fid = r.readULEB();
    uint64_t opid = r.readULEB();
    uint64_t fileid = r.readULEB();
    uint64_t sl = r.readULEB();
    uint64_t sc = r.readULEB();
    uint64_t el = r.readULEB();
    uint64_t ec = r.readULEB();
    di.locations.push_back({fid, opid, fileid, sl, sc, el, ec});
  }

  // OpSnippets
  uint64_t scnt = r.readULEB();
  di.snippets.reserve(scnt);
  for (uint64_t i = 0; i < scnt; ++i) {
    uint64_t fid = r.readULEB();
    uint64_t opid = r.readULEB();
    uint64_t ssid = r.readULEB();
    di.snippets.push_back({fid, opid, ssid});
  }

  if (r.p != r.end) throw std::runtime_error("trailing bytes in DEBUGINFO");
  return di;
}


static void parseTypesSection(const std::vector<uint8_t>& data,
                             const std::vector<std::string>& strings,
                             std::vector<TypeEntry>& types) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  types.clear();
  types.reserve(cnt + 1);
  types.push_back({0, ""});
  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    uint8_t flags = r.readU8();
    if ((flags & 0x1) == 0) throw std::runtime_error("type missing asm");
    uint64_t sid = r.readULEB();
    if (sid >= strings.size()) throw std::runtime_error("bad asm_sid");
    types.push_back({tag, strings[sid]});
  }
  if (r.p != r.end) throw std::runtime_error("trailing bytes in TYPES");
}

static void parseAttrsSection(const std::vector<uint8_t>& data,
                             const std::vector<std::string>& strings,
                             std::vector<AttrEntry>& attrs) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  attrs.clear();
  attrs.reserve(cnt + 1);
  attrs.push_back({0, ""});
  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    uint8_t flags = r.readU8();
    if ((flags & 0x1) == 0) throw std::runtime_error("attr missing asm");
    uint64_t sid = r.readULEB();
    if (sid >= strings.size()) throw std::runtime_error("bad asm_sid");
    attrs.push_back({tag, strings[sid]});
  }
  if (r.p != r.end) throw std::runtime_error("trailing bytes in ATTRS");
}

static void parseConstPoolSection(const std::vector<uint8_t>& data,
                                 std::vector<ConstEntryParsed>& consts) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  consts.clear();
  consts.reserve(cnt);

  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    if (tag == 0x01) {
      uint64_t tid = r.readULEB();
      int64_t v = r.readSLEB();
      ConstEntryParsed e;
      e.tag = tag;
      e.typeId = tid;
      e.intValue = v;
      consts.push_back(std::move(e));
    } else if (tag == 0x02) {
      uint64_t tid = r.readULEB();
      uint64_t blen = r.readULEB();
      auto bytes = r.readBytes(blen);
      ConstEntryParsed e;
      e.tag = tag;
      e.typeId = tid;
      e.floatBytes = std::move(bytes);
      consts.push_back(std::move(e));
    } else if (tag == 0x03) {
      // index vec (not yet needed for sample decoding)
      uint64_t n = r.readULEB();
      for (uint64_t j = 0; j < n; ++j) (void)r.readSLEB();
      ConstEntryParsed e;
      e.tag = tag;
      consts.push_back(std::move(e));
    } else if (tag == 0x04) {
      // wide int bits: type_id(uLEB), blen(uLEB), bytes(blen)
      uint64_t tid = r.readULEB();
      uint64_t blen = r.readULEB();
      auto bytes = r.readBytes(blen);
      ConstEntryParsed e;
      e.tag = tag;
      e.typeId = tid;
      e.intBytes = std::move(bytes);
      consts.push_back(std::move(e));
    } else {
      throw std::runtime_error("unknown ConstEntry tag");
    }
  }

  if (r.p != r.end) throw std::runtime_error("trailing bytes in CONSTPOOL");
}

struct BuildCtx {
  mlir::MLIRContext* ctx;
  const std::vector<std::string>* strings;
  const std::vector<TypeEntry>* types;
  const std::vector<AttrEntry>* attrs;
  const std::vector<ConstEntryParsed>* consts;

  // Function-global value_id table.
  std::vector<mlir::Value> values;

  // Function-global op_id table (preorder DFS).
  uint64_t* nextOpId = nullptr;
  std::vector<mlir::Operation*>* opsById = nullptr;
};

static mlir::Type getType(BuildCtx& bc, uint64_t tid) {
  if (tid >= bc.types->size()) throw std::runtime_error("bad type_id");
  return parseType(*bc.ctx, (*bc.types)[tid].asmStr);
}

static mlir::DictionaryAttr getAttrDict(BuildCtx& bc, uint64_t aid) {
  if (aid == 0) return mlir::DictionaryAttr::get(bc.ctx);
  if (aid >= bc.attrs->size()) throw std::runtime_error("bad attr_id");
  return parseAttrDict(*bc.ctx, (*bc.attrs)[aid].asmStr);
}

static void buildRegionInto(BuildCtx& bc, Reader& r, mlir::Region& region);

static mlir::Attribute buildConstAttr(BuildCtx &bc, uint64_t constId) {
  if (!bc.consts) throw std::runtime_error("constpool not available");
  if (constId >= bc.consts->size()) throw std::runtime_error("const_id out of range");
  const auto &e = (*bc.consts)[constId];

  if (e.tag == 0x01) {
    auto ty = getType(bc, e.typeId);
    auto it = mlir::dyn_cast<mlir::IntegerType>(ty);
    if (mlir::isa<mlir::IndexType>(ty)) {
      return mlir::IntegerAttr::get(ty, e.intValue);
    }
    if (!it) throw std::runtime_error("ConstInt type is not integer/index");
    return mlir::IntegerAttr::get(ty, e.intValue);
  }

  if (e.tag == 0x02) {
    auto ty = getType(bc, e.typeId);
    auto ft = mlir::dyn_cast<mlir::FloatType>(ty);
    if (!ft) throw std::runtime_error("ConstFloatBits type is not FloatType");

    unsigned bitWidth = ft.getWidth();
    unsigned byteLen = (bitWidth + 7) / 8;
    if (e.floatBytes.size() != byteLen) {
      throw std::runtime_error("ConstFloatBits byte_len mismatch");
    }

    const unsigned numWords = (bitWidth + 63) / 64;
    llvm::SmallVector<uint64_t, 4> words(numWords, 0);
    for (unsigned i = 0; i < byteLen; ++i) {
      unsigned w = i / 8;
      unsigned off = (i % 8) * 8;
      words[w] |= (uint64_t(e.floatBytes[i]) << off);
    }

    llvm::APInt bits(bitWidth, words);
    llvm::APFloat f(ft.getFloatSemantics(), bits);
    return mlir::FloatAttr::get(ft, f);
  }

  if (e.tag == 0x04) {
    auto ty = getType(bc, e.typeId);
    auto it = mlir::dyn_cast<mlir::IntegerType>(ty);
    if (!it) throw std::runtime_error("ConstIntBits type is not IntegerType");

    unsigned bitWidth = it.getWidth();
    unsigned byteLen = (bitWidth + 7) / 8;
    if (e.intBytes.size() != byteLen) {
      throw std::runtime_error("ConstIntBits byte_len mismatch");
    }

    const unsigned numWords = (bitWidth + 63) / 64;
    llvm::SmallVector<uint64_t, 4> words(numWords, 0);
    for (unsigned i = 0; i < byteLen; ++i) {
      unsigned w = i / 8;
      unsigned off = (i % 8) * 8;
      words[w] |= (uint64_t(e.intBytes[i]) << off);
    }

    llvm::APInt bits(bitWidth, words);
    return mlir::IntegerAttr::get(it, bits);
  }

  throw std::runtime_error("unsupported const tag");
}

static void buildOpList(BuildCtx& bc, Reader& r, mlir::Block& block) {
  const bool dbg = debugEnabled();
  uint64_t opcnt = r.readULEB();
  if (dbg) llvm::errs() << "[ptobc]   ops=" << opcnt << "\n";

  for (uint64_t oi = 0; oi < opcnt; ++oi) {
    if (dbg) llvm::errs() << "[ptobc]    op[" << oi << "]...\n";
    const uint64_t opId = bc.nextOpId ? (*bc.nextOpId)++ : 0;

    uint16_t opcode = r.readU16LE();
    uint64_t attrId = r.readULEB();

    // Generic escape.
    if (opcode == kOpcodeGeneric) {
      uint64_t nameSid = r.readULEB();
      if (nameSid >= bc.strings->size()) throw std::runtime_error("bad op_name sid");
      std::string opName = (*bc.strings)[nameSid];

      uint64_t nres = r.readULEB();
      llvm::SmallVector<mlir::Type, 4> resTypes;
      resTypes.reserve(nres);

      const size_t resStart = bc.values.size();
      for (uint64_t i = 0; i < nres; ++i) {
        uint64_t tid = r.readULEB();
        resTypes.push_back(getType(bc, tid));
        bc.values.push_back(mlir::Value());
      }

      uint64_t nops = r.readULEB();
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.reserve(nops);
      for (uint64_t i = 0; i < nops; ++i) {
        uint64_t vid = r.readULEB();
        if (vid >= bc.values.size()) throw std::runtime_error("operand value_id out of range");
        operands.push_back(bc.values[vid]);
      }

      uint64_t nreg = r.readULEB();

      mlir::OperationState st(mlir::UnknownLoc::get(bc.ctx), opName);
      st.addOperands(operands);
      st.addTypes(resTypes);

      auto dict = getAttrDict(bc, attrId);
      for (auto na : dict) {
        st.addAttribute(na.getName(), na.getValue());
      }

      for (uint64_t ri = 0; ri < nreg; ++ri) (void)st.addRegion();

      mlir::Operation* op = mlir::Operation::create(st);
      block.getOperations().push_back(op);

      if (bc.opsById) {
        if (opId >= bc.opsById->size()) bc.opsById->resize(opId + 1, nullptr);
        (*bc.opsById)[opId] = op;
      }

      for (uint64_t i = 0; i < nres; ++i) {
        bc.values[resStart + i] = op->getResult(i);
      }

      for (uint64_t ri = 0; ri < nreg; ++ri) {
        buildRegionInto(bc, r, op->getRegion(ri));
      }
      continue;
    }

    // Known compact op.
    const auto *info = ptobc::v0::lookupByOpcode(opcode);
    if (!info) throw std::runtime_error("missing opcode schema");

    uint8_t variant = 0;
    if (info->has_variant_u8) {
      variant = r.readU8();
    }

    // immediates
    uint8_t cmpPred = 0;
    uint8_t evA = 0, evB = 0, evC = 0;
    uint64_t constId = 0;
    uint8_t listMode = 0;
    uint64_t n1 = 0, n2 = 0;
    uint8_t optMask = 0;

    switch (info->imm_kind) {
      case 0x00:
        break;
      case 0x01:
        cmpPred = r.readU8();
        break;
      case 0x02:
        evA = r.readU8();
        evB = r.readU8();
        evC = r.readU8();
        break;
      case 0x05:
        constId = r.readULEB();
        break;
      case 0x06:
      case 0x07:
        listMode = r.readU8();
        n1 = r.readULEB();
        n2 = r.readULEB();
        break;
      case 0x08:
        optMask = r.readU8();
        break;
      default:
        throw std::runtime_error("unknown imm_kind");
    }

    auto readValueIds = [&](size_t n) {
      llvm::SmallVector<uint64_t, 8> ids;
      ids.reserve(n);
      for (size_t i = 0; i < n; ++i) ids.push_back(r.readULEB());
      return ids;
    };

    llvm::SmallVector<uint64_t, 8> operandIds;

    if (info->operand_mode == 0x00) {
      operandIds = readValueIds(info->num_operands);
    } else if (info->operand_mode == 0x01) {
      auto n = ptobc::v0::lookupOperandsByVariant(opcode, variant);
      if (!n) throw std::runtime_error("missing by-variant operand count");
      operandIds = readValueIds(*n);
    } else if (info->operand_mode == 0x02) {
      uint64_t n = r.readULEB();
      operandIds = readValueIds(n);
    } else if (info->operand_mode == 0x03) {
      if (listMode != 0) throw std::runtime_error("list_mode=1 not supported yet");
      size_t n = size_t(info->num_operands) + size_t(n1) + size_t(n2);
      operandIds = readValueIds(n);
    } else if (info->operand_mode == 0x04) {
      size_t n = ((optMask & 0x1) ? 1 : 0) + ((optMask & 0x2) ? 1 : 0);
      operandIds = readValueIds(n);
    } else {
      throw std::runtime_error("unknown operand_mode");
    }

    llvm::SmallVector<mlir::Value, 8> operands;
    operands.reserve(operandIds.size());
    for (auto vid : operandIds) {
      if (vid >= bc.values.size()) throw std::runtime_error("operand value_id out of range");
      operands.push_back(bc.values[vid]);
    }

    // result types
    llvm::SmallVector<mlir::Type, 4> resTypes;
    resTypes.reserve(info->num_results);
    if (info->result_type_mode == 0x01) {
      for (unsigned i = 0; i < info->num_results; ++i) {
        uint64_t tid = r.readULEB();
        resTypes.push_back(getType(bc, tid));
      }
    } else {
      // v0 currently expects explicit for all result-producing ops.
      for (unsigned i = 0; i < info->num_results; ++i) {
        resTypes.push_back(mlir::NoneType::get(bc.ctx));
      }
    }

    // Reserve value ids for results.
    const size_t resStart = bc.values.size();
    for (unsigned i = 0; i < info->num_results; ++i) {
      bc.values.push_back(mlir::Value());
    }

    // op name
    const char *opNameC = ptobc::v0::fullNameFromOpcodeVariant(opcode, variant);
    if (!opNameC) throw std::runtime_error("failed to map opcode->name");
    llvm::StringRef opName(opNameC);

    mlir::OperationState st(mlir::UnknownLoc::get(bc.ctx), opName);
    st.addOperands(operands);
    st.addTypes(resTypes);

    auto dict = getAttrDict(bc, attrId);
    for (auto na : dict) {
      st.addAttribute(na.getName(), na.getValue());
    }

    // immediate-derived attributes
    if (info->imm_kind == 0x01) {
      auto pred = (mlir::arith::CmpIPredicate)cmpPred;
      st.addAttribute("predicate", mlir::arith::CmpIPredicateAttr::get(bc.ctx, pred));
    } else if (info->imm_kind == 0x02) {
      st.addAttribute("src_op", mlir::pto::SyncOpTypeAttr::get(bc.ctx, (mlir::pto::SyncOpType)evA));
      st.addAttribute("dst_op", mlir::pto::SyncOpTypeAttr::get(bc.ctx, (mlir::pto::SyncOpType)evB));
      st.addAttribute("event_id", mlir::pto::EventAttr::get(bc.ctx, (mlir::pto::EVENT)evC));
    } else if (info->imm_kind == 0x05) {
      st.addAttribute("value", buildConstAttr(bc, constId));
    }

    // regions
    for (unsigned ri = 0; ri < info->num_regions; ++ri) (void)st.addRegion();

    mlir::Operation *op = mlir::Operation::create(st);
    block.getOperations().push_back(op);

    if (bc.opsById) {
      if (opId >= bc.opsById->size()) bc.opsById->resize(opId + 1, nullptr);
      (*bc.opsById)[opId] = op;
    }

    for (unsigned i = 0; i < info->num_results; ++i) {
      bc.values[resStart + i] = op->getResult(i);
    }

    for (unsigned ri = 0; ri < info->num_regions; ++ri) {
      buildRegionInto(bc, r, op->getRegion(ri));
    }
  }
}

static void buildRegionInto(BuildCtx& bc, Reader& r, mlir::Region& region) {
  const bool dbg = debugEnabled();
  uint64_t bcnt = r.readULEB();
  if (dbg) llvm::errs() << "[ptobc] region: blocks=" << bcnt << "\n";
  region.getBlocks().clear();

  for (uint64_t bi = 0; bi < bcnt; ++bi) {
    if (dbg) llvm::errs() << "[ptobc]  block[" << bi << "]...\n";
    auto* block = new mlir::Block();

    uint64_t nargs = r.readULEB();
    if (dbg) llvm::errs() << "[ptobc]   nargs=" << nargs << "\n";
    for (uint64_t ai = 0; ai < nargs; ++ai) {
      uint64_t tid = r.readULEB();
      auto ty = getType(bc, tid);
      auto arg = block->addArgument(ty, mlir::UnknownLoc::get(bc.ctx));
      bc.values.push_back(arg);
    }

    buildOpList(bc, r, *block);
    region.push_back(block);
  }
}

static mlir::ModuleOp decodeToModule(mlir::MLIRContext& ctx,
                                    const std::vector<std::string>& strings,
                                    const std::vector<TypeEntry>& types,
                                    const std::vector<AttrEntry>& attrs,
                                    const std::vector<uint8_t>& constPool,
                                    const std::vector<uint8_t>& moduleBytes,
                                    std::vector<std::vector<mlir::Operation*>>* opsByFuncOut) {
  const bool dbg = debugEnabled();

  Reader r{moduleBytes.data(), moduleBytes.data() + moduleBytes.size()};
  uint8_t profile = r.readU8();
  uint8_t indexWidth = r.readU8();
  if (dbg) llvm::errs() << "[ptobc] module: profile=" << unsigned(profile) << " indexWidth=" << unsigned(indexWidth) << "\n";

  uint64_t moduleAttrId = r.readULEB();
  uint64_t gcnt = r.readULEB();
  if (dbg) llvm::errs() << "[ptobc] module: moduleAttrId=" << moduleAttrId << " globals=" << gcnt << "\n";
  for (uint64_t i = 0; i < gcnt; ++i) {
    throw std::runtime_error("globals not supported");
  }

  uint64_t fcnt = r.readULEB();
  if (dbg) llvm::errs() << "[ptobc] module: funcs=" << fcnt << "\n";

  struct FuncDecl { std::string name; mlir::FunctionType type; mlir::DictionaryAttr attrs; uint8_t flags; };
  std::vector<FuncDecl> decls;
  decls.reserve(fcnt);

  std::vector<ConstEntryParsed> consts;
  parseConstPoolSection(constPool, consts);

  BuildCtx bc{&ctx, &strings, &types, &attrs, &consts, {}, nullptr, nullptr};

  for (uint64_t i = 0; i < fcnt; ++i) {
    uint64_t nameSid = r.readULEB();
    uint64_t ftypeId = r.readULEB();
    uint8_t flags = r.readU8();
    uint64_t fattrId = r.readULEB();
    if (nameSid >= strings.size()) throw std::runtime_error("bad func name sid");
    if (ftypeId >= types.size()) throw std::runtime_error("bad func type id");

    if (dbg) llvm::errs() << "[ptobc] func[" << i << "]: nameSid=" << nameSid << " ftypeId=" << ftypeId << " flags=" << unsigned(flags) << " fattrId=" << fattrId << "\n";

    auto ty = parseType(ctx, types.at(ftypeId).asmStr);
    auto fty = mlir::dyn_cast<mlir::FunctionType>(ty);
    if (!fty) throw std::runtime_error("func type parse failed");

    decls.push_back({strings[nameSid], fty, getAttrDict(bc, fattrId), flags});
  }

  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  // Apply module attrs
  auto modDict = getAttrDict(bc, moduleAttrId);
  for (auto na : modDict) {
    module->setAttr(na.getName(), na.getValue());
  }

  for (uint64_t i = 0; i < fcnt; ++i) {
    if (dbg) llvm::errs() << "[ptobc] building func body: " << decls[i].name << "\n";

    auto fn = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&ctx), decls[i].name, decls[i].type);
    if (dbg) llvm::errs() << "[ptobc] created func op\n";
    for (auto na : decls[i].attrs) {
      fn->setAttr(na.getName(), na.getValue());
    }

    if ((decls[i].flags & 0x1) == 0) {
      // decode body region
      bc.values.clear();

      uint64_t nextOpId = 0;
      std::vector<mlir::Operation*> opsById;
      bc.nextOpId = &nextOpId;
      bc.opsById = &opsById;

      buildRegionInto(bc, r, fn.getBody());
      if (dbg) llvm::errs() << "[ptobc] func body built ok: values=" << bc.values.size() << " ops=" << opsById.size() << "\n";

      if (opsByFuncOut) opsByFuncOut->push_back(std::move(opsById));
    } else {
      if (opsByFuncOut) opsByFuncOut->push_back({});
    }

    module.push_back(fn);
  }

  if (r.p != r.end) throw std::runtime_error("trailing bytes in MODULE");
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp>
decodePTOBCToModule(llvm::ArrayRef<uint8_t> fileBytes, mlir::MLIRContext &ctx) {
  const bool dbg = debugEnabled();

  if (fileBytes.size() < 14) throw std::runtime_error("file too small");
  if (std::memcmp(fileBytes.data(), "PTOBC\0", 6) != 0) throw std::runtime_error("bad magic");

  uint16_t ver = uint16_t(fileBytes[6]) | (uint16_t(fileBytes[7]) << 8);
  if (ver != kVersionV0) throw std::runtime_error("unsupported version");

  uint32_t payloadLen = uint32_t(fileBytes[10]) | (uint32_t(fileBytes[11]) << 8) | (uint32_t(fileBytes[12]) << 16) | (uint32_t(fileBytes[13]) << 24);
  if (payloadLen != fileBytes.size() - 14) throw std::runtime_error("payload_len mismatch");

  Reader r{fileBytes.data() + 14, fileBytes.data() + fileBytes.size()};

  auto readSection = [&]() -> std::pair<uint8_t, std::vector<uint8_t>> {
    uint8_t sid = r.readU8();
    uint32_t slen = r.readU32LE();
    auto bytes = r.readBytes(slen);
    if (dbg) llvm::errs() << "[ptobc] section id=" << unsigned(sid) << " len=" << slen << "\n";
    return {sid, bytes};
  };

  auto [s1, d1] = readSection();
  auto [s2, d2] = readSection();
  auto [s3, d3] = readSection();
  auto [s4, d4] = readSection();
  auto [s6, d6] = readSection();

  std::optional<DebugInfo> dbgInfo;
  // Optional trailing sections: DEBUGINFO, EXTRA.
  while (r.p != r.end) {
    auto [sid, sec] = readSection();
    if (sid == kSectionDebugInfo) {
      if (dbgInfo) throw std::runtime_error("duplicate DEBUGINFO section");
      dbgInfo = parseDebugInfoSection(sec);
    } else if (sid == kSectionExtra) {
      // Ignore EXTRA payload for now.
    } else {
      throw std::runtime_error("unexpected trailing section id");
    }
  }

  if (s1 != kSectionStrings || s2 != kSectionTypes || s3 != kSectionAttrs || s4 != kSectionConstPool || s6 != kSectionModule) {
    throw std::runtime_error("unexpected section order");
  }

  std::vector<std::string> strings;
  parseStringsSection(d1, strings);

  std::vector<TypeEntry> types;
  parseTypesSection(d2, strings, types);

  std::vector<AttrEntry> attrs;
  parseAttrsSection(d3, strings, attrs);

  if (dbg) {
    llvm::errs() << "[ptobc] strings=" << strings.size() << " types=" << types.size() << " attrs=" << attrs.size() << " moduleBytes=" << d6.size() << "\n";
  }

  // Ensure dialects are loaded before we start materializing ops.
  (void)ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  (void)ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  (void)ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  (void)ctx.getOrLoadDialect<mlir::pto::PTODialect>();

  if (dbg) llvm::errs() << "[ptobc] decoding module...\n";

  std::vector<std::vector<mlir::Operation*>> opsByFunc;
  auto module = decodeToModule(ctx, strings, types, attrs, d4, d6, dbgInfo ? &opsByFunc : nullptr);

  // Apply op locations from DEBUGINFO (best-effort).
  if (dbgInfo) {
    for (const auto &l : dbgInfo->locations) {
      if (l.funcId >= opsByFunc.size()) continue;
      auto &ops = opsByFunc[l.funcId];
      if (l.opId >= ops.size()) continue;
      mlir::Operation *op = ops[l.opId];
      if (!op) continue;
      if (l.fileId >= dbgInfo->files.size()) continue;
      const auto &f = dbgInfo->files[l.fileId];
      if (f.pathSid >= strings.size()) continue;

      auto path = strings[f.pathSid];
      auto loc = mlir::FileLineColLoc::get(&ctx, path, (unsigned)l.sl, (unsigned)l.sc);
      op->setLoc(loc);
    }
  }

  return module;
}

void decodeFileToPTO(const std::string& inPath, const std::string& outPath) {
  const bool dbg = debugEnabled();

  if (dbg) llvm::errs() << "[ptobc] decode: reading file: " << inPath << "\n";
  auto data = readFile(inPath);

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::pto::PTODialect>();
  mlir::MLIRContext ctx(registry);
  ctx.allowUnregisteredDialects(true);

  auto module = decodePTOBCToModule(data, ctx);

  if (dbg) llvm::errs() << "[ptobc] decoded module ok; printing...\n";

  CanonicalPrintOptions opt;
  opt.generic = (std::getenv("PTOBC_PRINT_GENERIC") != nullptr);
  opt.keepMLIRFloatPrinting = (std::getenv("PTOBC_PRINT_PRETTY") != nullptr);
  opt.printDebugInfo = (std::getenv("PTOBC_PRINT_LOC") != nullptr);

  std::string out = printModuleCanonical(module.get(), opt);

  if (dbg) llvm::errs() << "[ptobc] writing output: " << outPath << "\n";
  std::ofstream ofs(outPath);
  ofs << out;
  if (!out.empty() && out.back() != '\n') ofs << "\n";
}

} // namespace ptobc
