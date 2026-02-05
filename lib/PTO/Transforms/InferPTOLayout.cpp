//===- InferPTOLayout.cpp - Infer layout for make_tensor_view -------------===//
//
// Keeps layout as an optional attribute. If shape/stride are fully static,
// infer ND/DN/NZ and attach `layout`; otherwise leave it unset.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<int64_t> getConstIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static unsigned elemByteSize(Type ty) {
  if (auto f = dyn_cast<FloatType>(ty))
    return f.getWidth() / 8;
  if (auto i = dyn_cast<IntegerType>(ty))
    return i.getWidth() / 8;
  return 0;
}

static std::optional<Layout> inferLayout(ArrayRef<int64_t> shape,
                                         ArrayRef<int64_t> strides,
                                         unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;

  // NZ: rank>=5, sh3 == 16, sh3*sh4*elemBytes == 512, st5==1, st4==sh5
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return Layout::NZ;
  }

  bool isRow = strides.back() == 1 &&
               llvm::all_of(llvm::zip(strides.drop_back(), strides.drop_front(),
                                      shape.drop_front()),
                            [](auto tup) {
                              auto s0 = std::get<0>(tup);
                              auto s1 = std::get<1>(tup);
                              auto sh = std::get<2>(tup);
                              return s0 == s1 * sh;
                            });
  if (isRow)
    return Layout::ND;

  bool isCol = strides.front() == 1 &&
               llvm::all_of(llvm::zip(strides.drop_front(), strides.drop_back(),
                                      shape.drop_back()),
                            [](auto tup) {
                              auto s1 = std::get<0>(tup);
                              auto s0 = std::get<1>(tup);
                              auto sh = std::get<2>(tup);
                              return s1 == s0 * sh;
                            });
  if (isCol)
    return Layout::DN;

  return Layout::ND; // fallback
}

struct InferPTOLayoutPass
    : public PassWrapper<InferPTOLayoutPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferPTOLayoutPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](MakeTensorViewOp op) {
      if (op.getLayoutAttr())
        return;

      auto tvTy = dyn_cast<TensorViewType>(op.getResult().getType());
      if (!tvTy)
        return;

      // If shape/stride is dynamic, fill default ND and warn here (avoid
      // warnings in the verifier to sidestep recursive verify-on-print).
      if (llvm::any_of(tvTy.getShape(),
                       [](int64_t v) { return v == ShapedType::kDynamic; })) {
        op.emitWarning() << "dynamic shape without explicit 'layout'; "
                         << "defaulting to ND";
        op->setAttr("layout", LayoutAttr::get(op.getContext(), Layout::ND));
        return;
      }

      SmallVector<int64_t> strides;
      strides.reserve(op.getStrides().size());
      bool allConstStride = true;
      for (Value s : op.getStrides()) {
        auto v = getConstIndex(s);
        if (!v) {
          allConstStride = false;
          break;
        }
        strides.push_back(*v);
      }
      if (!allConstStride) {
        op.emitWarning() << "dynamic stride without explicit 'layout'; "
                         << "defaulting to ND";
        op->setAttr("layout", LayoutAttr::get(op.getContext(), Layout::ND));
        return;
      }

      SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
      if (auto layout =
              inferLayout(shape, strides, elemByteSize(tvTy.getElementType()))) {
        op->setAttr("layout", LayoutAttr::get(op.getContext(), *layout));
      } else {
        // Fallback ND if we cannot infer (should be rare with static info).
        op->setAttr("layout", LayoutAttr::get(op.getContext(), Layout::ND));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOLayoutPass() {
  return std::make_unique<InferPTOLayoutPass>();
}
