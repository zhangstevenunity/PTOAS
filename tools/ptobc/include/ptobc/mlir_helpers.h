#pragma once

#include <string>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

namespace ptobc {

std::string printType(mlir::Type t);
std::string printAttr(mlir::Attribute a);
std::string printAttrDict(mlir::DictionaryAttr a);

mlir::Type parseType(mlir::MLIRContext& ctx, const std::string& s);
mlir::Attribute parseAttr(mlir::MLIRContext& ctx, const std::string& s);
mlir::DictionaryAttr parseAttrDict(mlir::MLIRContext& ctx, const std::string& s);

} // namespace ptobc
