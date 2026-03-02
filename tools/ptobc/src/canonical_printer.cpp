#include "ptobc/canonical_printer.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ptobc {

static std::vector<std::string> splitLinesPreserveEmpty(const std::string &s) {
  std::vector<std::string> lines;
  std::string cur;
  for (char c : s) {
    if (c == '\n') {
      lines.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  // If the string ended with '\n', cur is empty and represents the final
  // trailing empty line. We do NOT keep that in lines; the join below will
  // re-add newlines.
  lines.push_back(cur);
  return lines;
}

static std::string joinLines(const std::vector<std::string> &lines) {
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    out += lines[i];
    if (i + 1 < lines.size()) out.push_back('\n');
  }
  return out;
}

static void stripUnknownLocSuffix(std::vector<std::string> &lines) {
  // MLIR prints `loc(unknown)` when debug printing is enabled. It is pure noise
  // for canonical output, so strip it.
  static const std::regex re(R"(^(.*?)(?:[ \t]+loc\(unknown\))[ \t]*$)");
  for (auto &ln : lines) {
    std::smatch m;
    if (std::regex_match(ln, m, re)) {
      ln = m[1].str();
    }
  }
}

static std::string hexFloatLiteral(mlir::FloatAttr a) {
  llvm::SmallString<32> s;
  llvm::raw_svector_ostream os(s);
  llvm::SmallVector<char, 32> digits;
  llvm::APInt bits = a.getValue().bitcastToAPInt();
  bits.toString(digits, /*Radix=*/16, /*Signed=*/false, /*formatAsCLiteral=*/true);
  os << llvm::StringRef(digits.data(), digits.size());
  return os.str().str();
}

static void sortAttributesLexicographically(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation *op) {
    auto attrs = op->getAttrs();
    if (attrs.size() <= 1) return;

    llvm::SmallVector<mlir::NamedAttribute, 8> sorted(attrs.begin(), attrs.end());
    llvm::sort(sorted, [](const mlir::NamedAttribute &a, const mlir::NamedAttribute &b) {
      return a.getName().getValue() < b.getName().getValue();
    });

    // Only write back if something actually changed.
    if (llvm::equal(sorted, attrs, [](const mlir::NamedAttribute &x, const mlir::NamedAttribute &y) {
          return x.getName() == y.getName() && x.getValue() == y.getValue();
        })) {
      return;
    }

    op->setAttrs(sorted);
  });
}

static bool isSSAIdentChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         c == '_' || c == '.' || c == '$' || c == '-' ;
}

static std::string renameSSAInText(const std::string &text,
                                   const std::unordered_map<std::string, std::string> &map) {
  std::string out;
  out.reserve(text.size());

  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (c != '%') {
      out.push_back(c);
      continue;
    }

    // Parse SSA identifier after '%'.
    size_t j = i + 1;
    while (j < text.size() && isSSAIdentChar(text[j])) ++j;
    if (j == i + 1) {
      out.push_back('%');
      continue;
    }

    std::string key = text.substr(i + 1, j - (i + 1));
    auto it = map.find(key);
    if (it == map.end()) {
      // No rename.
      out.push_back('%');
      out.append(key);
    } else {
      out.push_back('%');
      out.append(it->second);
    }

    i = j - 1;
  }
  return out;
}

static void collectSSADefsFromSignature(const std::string &line,
                                        std::vector<std::string> &out) {
  // Collect `%name:` occurrences within (...) in `func.func` or `^bb` lines.
  // This is a lightweight heuristic but works for standard MLIR assembly.
  size_t lpar = line.find('(');
  if (lpar == std::string::npos) return;
  size_t rpar = line.find(')', lpar + 1);
  if (rpar == std::string::npos) return;

  for (size_t i = lpar + 1; i < rpar; ++i) {
    if (line[i] != '%') continue;
    size_t j = i + 1;
    while (j < rpar && isSSAIdentChar(line[j])) ++j;
    if (j == i + 1) continue;
    // Must be followed by ':' to be an arg/blkarg.
    if (j < rpar && line[j] == ':') {
      out.push_back(line.substr(i + 1, j - (i + 1)));
    }
    i = j;
  }
}

static bool parseConstantLine(const std::string &line, std::string &imm, std::string &ty) {
  // Match: "%x = arith.constant <imm> : <ty> [loc(...)]" (pretty form)
  // We don't try to parse attributes on constants.
  static const std::regex re(R"(^[ \t]*%[-a-zA-Z$._0-9]+[ \t]*=[ \t]*arith\.constant[ \t]+(.+?)[ \t]*:[ \t]*([^ \t]+)(?:[ \t]+loc\(.+\))?[ \t]*$)");
  std::smatch m;
  if (!std::regex_match(line, m, re)) return false;
  imm = m[1].str();
  ty = m[2].str();
  return true;
}

static std::string canonicalConstBaseName(const std::string &imm, const std::string &ty) {
  // Build a deterministic `%c...`-style base name derived from the printed immediate.
  // Examples:
  //   imm=32 ty=index -> c32
  //   imm=7 ty=i32 -> c7_i32
  //   imm=0x3F800000 ty=f32 -> c0x3F800000_f32
  std::string base = "c";
  base += imm;
  if (ty != "index") {
    base += "_";
    base += ty;
  }
  return base;
}

static std::string canonicalizeSSANames(const std::string &printed) {
  auto lines = splitLinesPreserveEmpty(printed);

  std::vector<std::string> defs;
  defs.reserve(256);

  for (const auto &ln : lines) {
    if (ln.find("func.func") != std::string::npos) {
      collectSSADefsFromSignature(ln, defs);
      continue;
    }
    if (!ln.empty() && ln[0] == '^') {
      collectSSADefsFromSignature(ln, defs);
      continue;
    }
    // Op results at start of line.
    // e.g. "  %12 = ..." or "  %c0 = ..." or "%0:2 = ...".
    size_t pos = ln.find('%');
    if (pos == std::string::npos) continue;
    // Require this to be a definition: '%' must appear before '='.
    size_t eq = ln.find('=');
    if (eq == std::string::npos || pos > eq) continue;

    size_t j = pos + 1;
    while (j < ln.size() && isSSAIdentChar(ln[j])) ++j;
    if (j == pos + 1) continue;
    defs.push_back(ln.substr(pos + 1, j - (pos + 1)));
  }

  std::unordered_map<std::string, std::string> ren;
  ren.reserve(defs.size() * 2);

  // Pre-scan constants for nicer `%c...` aliases.
  std::unordered_map<std::string, int> constCounts;

  // Assign names in definition order, but keep constants named via their immediates.
  uint64_t nextNonConst = 0;

  for (const auto &old : defs) {
    // Find the line that defines this value to see if it is a constant.
    // (Linear scan; ok for now.)
    bool isConst = false;
    std::string imm, ty;
    for (const auto &ln : lines) {
      // quick filter
      if (ln.find('%' + old) == std::string::npos) continue;
      if (ln.find("= arith.constant") == std::string::npos) continue;
      // Must be the definition line.
      size_t pos = ln.find('%');
      if (pos == std::string::npos) continue;
      size_t j = pos + 1;
      while (j < ln.size() && isSSAIdentChar(ln[j])) ++j;
      if (ln.substr(pos + 1, j - (pos + 1)) != old) continue;
      if (parseConstantLine(ln, imm, ty)) {
        isConst = true;
      }
      break;
    }

    if (isConst) {
      std::string base = canonicalConstBaseName(imm, ty);
      int &n = constCounts[base];
      std::string name = base;
      if (n > 0) name += "_" + std::to_string(n);
      ++n;
      ren.emplace(old, name);
    } else {
      ren.emplace(old, std::to_string(nextNonConst++));
    }
  }

  return renameSSAInText(printed, ren);
}

static void canonicalizeScalarFloatConstants(mlir::ModuleOp module,
                                             const mlir::AsmState::LocationMap &locMap,
                                             std::vector<std::string> &lines) {
  // Match: "%name = arith.constant <anything> : fXX [loc(...)]"
  // We keep the prefix + the type suffix (+ optional loc suffix), replace the literal.
  const std::regex re(R"(^([ \t]*%[-a-zA-Z$._0-9]+[ \t]*=[ \t]*arith\.constant[ \t]+)(.+?)([ \t]*:[ \t]*f(16|32|64)(?:[ \t]+loc\(.+\))?[ \t]*$))");

  module.walk([&](mlir::Operation *op) {
    auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(op);
    if (!cst) return;

    auto f = llvm::dyn_cast<mlir::FloatAttr>(cst.getValue());
    if (!f) return;

    // Only canonicalize scalar float constants.
    if (!llvm::isa<mlir::FloatType>(cst.getType())) return;

    auto it = locMap.find(op);
    if (it == locMap.end()) return;

    unsigned lineNo = it->second.first;
    if (lineNo == 0) return;
    size_t idx = size_t(lineNo - 1);
    if (idx >= lines.size()) return;

    std::smatch m;
    if (!std::regex_match(lines[idx], m, re)) {
      // If the format doesn't match (e.g., multi-result or dialect-specific
      // printer changes), leave it as-is.
      return;
    }

    std::string lit = hexFloatLiteral(f);
    lines[idx] = m[1].str() + lit + m[3].str();
  });
}

std::string printModuleCanonical(mlir::ModuleOp module,
                                 const CanonicalPrintOptions &opt) {
  // Enforce canonical attribute ordering before printing.
  sortAttributesLexicographically(module);

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  flags.assumeVerified();
  if (opt.generic) flags.printGenericOpForm();
  if (opt.printDebugInfo) flags.enableDebugInfo(true, /*prettyForm=*/false);

  mlir::AsmState::LocationMap locMap;
  mlir::AsmState state(module.getOperation(), flags, &locMap);

  std::string printed;
  llvm::raw_string_ostream os(printed);
  module.getOperation()->print(os, state);
  os.flush();

  if (opt.keepMLIRFloatPrinting) {
    return printed;
  }

  // Canonicalize floats in-place.
  auto lines = splitLinesPreserveEmpty(printed);
  canonicalizeScalarFloatConstants(module, locMap, lines);

  // If debug printing is enabled, strip noise like `loc(unknown)`.
  stripUnknownLocSuffix(lines);

  std::string out = joinLines(lines);

  // Canonicalize SSA naming.
  out = canonicalizeSSANames(out);

  return out;
}

} // namespace ptobc
