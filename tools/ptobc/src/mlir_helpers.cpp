#include "ptobc/mlir_helpers.h"

#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/Support/raw_ostream.h>

#include <stdexcept>

namespace ptobc {

std::string printType(mlir::Type t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  t.print(os);
  os.flush();
  return s;
}

std::string printAttr(mlir::Attribute a) {
  std::string s;
  llvm::raw_string_ostream os(s);
  a.print(os);
  os.flush();
  return s;
}

std::string printAttrDict(mlir::DictionaryAttr a) {
  return printAttr(a);
}

mlir::Type parseType(mlir::MLIRContext& ctx, const std::string& s) {
  mlir::Type t = mlir::parseType(s, &ctx);
  if (!t) throw std::runtime_error("failed to parse type: " + s);
  return t;
}

mlir::Attribute parseAttr(mlir::MLIRContext& ctx, const std::string& s) {
  mlir::Attribute a = mlir::parseAttribute(s, &ctx);
  if (!a) throw std::runtime_error("failed to parse attr: " + s);
  return a;
}

mlir::DictionaryAttr parseAttrDict(mlir::MLIRContext& ctx, const std::string& s) {
  auto a = parseAttr(ctx, s);
  auto d = mlir::dyn_cast<mlir::DictionaryAttr>(a);
  if (!d) throw std::runtime_error("attr is not a dictionary: " + s);
  return d;
}

} // namespace ptobc
