//===- PTOAttrs.cpp ------------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"          // parseAttribute
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;

TileBufConfigAttr TileBufConfigAttr::getDefault(MLIRContext *ctx) {
  Builder b(ctx);
  BLayoutAttr bl = BLayoutAttr::get(ctx, BLayout::RowMajor);
  SLayoutAttr sl = SLayoutAttr::get(ctx, SLayout::NoneBox);
  PadValueAttr pv = PadValueAttr::get(ctx, PadValue::Null);
  IntegerAttr sz = b.getI32IntegerAttr(512);
  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv);
}

bool TileBufConfigAttr::isDefault() const {
  auto d = getDefault(getContext());
  return getBLayout() == d.getBLayout() &&
         getSLayout() == d.getSLayout() &&
         getSFractalSize() == d.getSFractalSize() &&
         getPad() == d.getPad();
}

static int32_t getLayoutInt(Attribute a, int32_t def) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) return static_cast<int32_t>(bl.getValue());
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) return static_cast<int32_t>(sl.getValue());
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) return static_cast<int32_t>(pv.getValue());
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return static_cast<int32_t>(ia.getInt());
  return def;
}

LogicalResult TileBufConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                       Attribute bLayout,
                                       Attribute sLayout,
                                       IntegerAttr sFractalSize,
                                       Attribute pad) {
  if (!bLayout || (!mlir::isa<BLayoutAttr>(bLayout) && !mlir::isa<IntegerAttr>(bLayout)))
    return emitError() << "blayout must be BLayoutAttr or i32 integer attr", failure();
  if (!sLayout || (!mlir::isa<SLayoutAttr>(sLayout) && !mlir::isa<IntegerAttr>(sLayout)))
    return emitError() << "slayout must be SLayoutAttr or i32 integer attr", failure();
  if (!pad || (!mlir::isa<PadValueAttr>(pad) && !mlir::isa<IntegerAttr>(pad)))
    return emitError() << "pad must be PadValueAttr or i32 integer attr", failure();

  if (!sFractalSize || !sFractalSize.getType().isInteger(32))
    return emitError() << "s_fractal_size must be i32", failure();

  int32_t s = (int32_t)sFractalSize.getInt();
  if (s != 32 && s != 16 && s != 512 && s != 1024)
    return emitError() << "unsupported s_fractal_size: " << s, failure();

  int32_t blv = getLayoutInt(bLayout, -1);
  if (blv != 0 && blv != 1)
    return emitError() << "unsupported blayout value: " << blv, failure();

  int32_t slv = getLayoutInt(sLayout, -1);
  if (slv < 0 || slv > 2)
    return emitError() << "unsupported slayout value: " << slv, failure();

  int32_t pvv = getLayoutInt(pad, -1);
  if (pvv < 0 || pvv > 3)
    return emitError() << "unsupported pad value: " << pvv, failure();

  return success();
}

// Helper: parse Attribute and convert to BLayoutAttr/SLayoutAttr/PadValueAttr
static BLayoutAttr toBLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a)) return bl;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return BLayoutAttr::get(ctx, static_cast<BLayout>(ia.getInt()));
  return {};
}
static SLayoutAttr toSLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a)) return sl;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return SLayoutAttr::get(ctx, static_cast<SLayout>(ia.getInt()));
  return {};
}
static PadValueAttr toPadValueAttr(MLIRContext *ctx, Attribute a) {
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a)) return pv;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a)) return PadValueAttr::get(ctx, static_cast<PadValue>(ia.getInt()));
  return {};
}

Attribute TileBufConfigAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto def = TileBufConfigAttr::getDefault(ctx);
  BLayoutAttr bl = def.getBLayout();
  SLayoutAttr sl = def.getSLayout();
  IntegerAttr sz = def.getSFractalSize();
  PadValueAttr pv = def.getPad();

  if (p.parseLess()) return {};

  if (succeeded(p.parseOptionalGreater()))
    return TileBufConfigAttr::get(ctx, bl, sl, sz, pv);

  while (true) {
    StringRef key;
    if (p.parseKeyword(&key)) return {};
    if (p.parseEqual()) return {};

    if (key == "blayout") {
      Attribute a;
      if (p.parseAttribute(a)) return {};
      bl = toBLayoutAttr(ctx, a);
      if (!bl) return {};
    } else if (key == "slayout") {
      Attribute a;
      if (p.parseAttribute(a)) return {};
      sl = toSLayoutAttr(ctx, a);
      if (!sl) return {};
    } else if (key == "s_fractal_size") {
      int32_t v;
      if (p.parseInteger(v)) return {};
      sz = IntegerAttr::get(IntegerType::get(ctx, 32), v);
    } else if (key == "pad") {
      Attribute a;
      if (p.parseAttribute(a)) return {};
      pv = toPadValueAttr(ctx, a);
      if (!pv) return {};
    } else {
      p.emitError(p.getCurrentLocation(), "unknown key in tile_buf_config: ") << key;
      return {};
    }

    if (succeeded(p.parseOptionalGreater()))
      break;
    if (p.parseComma()) return {};
  }

  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv);
}

void TileBufConfigAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "blayout=" << getBLayout();
  p << ", slayout=" << getSLayout();
  p << ", s_fractal_size=" << (int32_t)getSFractalSize().getInt();
  p << ", pad=" << getPad();
  p << ">";
}
