//===- PTO.h - PTO Dialect --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dialect for the PTO Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTO_H_
#define MLIR_DIALECT_PTO_IR_PTO_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// PTO Dialect
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTODialect.h"

//===----------------------------------------------------------------------===//
// PTO Enums
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOEnums.h.inc"

//===----------------------------------------------------------------------===//
// PTO Interfaces
//===----------------------------------------------------------------------===//
 
#include "PTO/IR/PTOInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// PTO Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.h.inc"

//===----------------------------------------------------------------------===//
// PTO Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.h.inc"

//===----------------------------------------------------------------------===//
// PTO Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.h.inc"

namespace mlir {
class TypeConverter;

namespace pto {

/// Get PTO Address Space Attr from input type.
AddressSpaceAttr getPTOAddressSpaceAttr(Type type);

/// Return true if type is a ptr/memref in GM address space (or default).
bool isScalarPtrOrMemRef(Type type);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTO_H_
