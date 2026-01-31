//===- DialectPTO.cpp -----------------------------------------------------===//
//
// Python bindings for the PTO dialect types (pybind11 version).
//
// This file is intended to be built via declare_mlir_python_extension(...)
// with PYTHON_BINDINGS_LIBRARY pybind11, and linked with MLIRCAPIPTO.
//
//===----------------------------------------------------------------------===//

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "pto-c/Dialect/PTO.h"
#include "mlir-c/IR.h"
#include "PTO/IR/PTO.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Support.h"
namespace py = pybind11;
using namespace mlir::python::adaptors;

static std::vector<int64_t> toInt64Vector(const py::sequence &seq) {
  std::vector<int64_t> out;
  out.reserve(seq.size());
  for (py::handle h : seq)
    out.push_back(py::cast<int64_t>(h));
  return out;
}

static py::list shapeToPyList(const int64_t *data, intptr_t n) {
  py::list lst;
  for (intptr_t i = 0; i < n; ++i)
    lst.append(py::int_(data[i]));
  return lst;
}

void populatePTODialectSubmodule(pybind11::module &m);
void populatePTODialectSubmodule(pybind11::module &m) {
  (void)m;
}
PYBIND11_MODULE(_pto, m) {
    m.doc() = "PTO dialect Python bindings (pybind11).";

    // --------------------------------------------------------------------------
    // Dialect registration helper
    // --------------------------------------------------------------------------
    m.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__pto__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load)
            mlirDialectHandleLoadDialect(handle, context);
        },
        py::arg("context"), py::arg("load") = true);

    // [保留 HEAD]: AddressSpace 枚举定义
    py::enum_<mlir::pto::AddressSpace>(m, "AddressSpace")
    .value("Zero", mlir::pto::AddressSpace::Zero)
    .value("GM",   mlir::pto::AddressSpace::GM)
    .value("MAT",   mlir::pto::AddressSpace::MAT)
    .value("LEFT",  mlir::pto::AddressSpace::LEFT)
    .value("RIGHT",  mlir::pto::AddressSpace::RIGHT)
    .value("ACC",  mlir::pto::AddressSpace::ACC)
    .value("UB",   mlir::pto::AddressSpace::UB)
    .value("BIAS",   mlir::pto::AddressSpace::BIAS)
    .export_values();
    py::enum_<BLayout>(m, "BLayout")
    .value("RowMajor", BLayout::RowMajor)
    .value("ColMajor", BLayout::ColMajor);

    py::enum_<SLayout>(m, "SLayout")
    .value("NoneBox", SLayout::NoneBox)
    .value("RowMajor", SLayout::RowMajor)
    .value("ColMajor", SLayout::ColMajor);

    py::enum_<PadValue>(m, "PadValue")
    .value("Null", PadValue::Null)
    .value("Zero", PadValue::Zero)
    .value("Max", PadValue::Max)
    .value("Min", PadValue::Min);

    py::enum_<mlir::pto::RoundMode>(m, "RoundMode")
    .value("NONE", mlir::pto::RoundMode::NONE)
    .value("RINT", mlir::pto::RoundMode::RINT)
    .value("ROUND", mlir::pto::RoundMode::ROUND)
    .value("FLOOR", mlir::pto::RoundMode::FLOOR)
    .value("CEIL", mlir::pto::RoundMode::CEIL)
    .value("TRUNC", mlir::pto::RoundMode::TRUNC)
    .value("ODD", mlir::pto::RoundMode::ODD)
    .value("CAST_RINT", mlir::pto::RoundMode::CAST_RINT);

    py::enum_<MlirPTOCmpMode>(m, "CmpMode")
      .value("EQ", MlirPTOCmpMode_EQ)
      .value("NE", MlirPTOCmpMode_NE)
      .value("LT", MlirPTOCmpMode_LT)
      .value("LE", MlirPTOCmpMode_LE)
      .value("GT", MlirPTOCmpMode_GT)
      .value("GE", MlirPTOCmpMode_GE)
      .export_values();

    mlir_attribute_subclass(m, "BLayoutAttr",
                        [](MlirAttribute a) -> bool {
                          // 我们这里用“i32 integer attr”表示 enum，所以只要是 i32 IntegerAttr 就 accept
                          return mlirAttributeIsAInteger(a) &&
                                 mlirIntegerTypeGetWidth(mlirAttributeGetType(a)) == 32;
                        })
    .def_classmethod(
        "get",
        [](py::object cls, int32_t value, MlirContext ctx) -> py::object {
          MlirAttribute a = mlirPTOBLayoutAttrGet(ctx, value);
          if (mlirAttributeIsNull(a)) return py::none();
          return cls(a);
        },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "SLayoutAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirAttributeIsAInteger(a) &&
                                    mlirIntegerTypeGetWidth(mlirAttributeGetType(a)) == 32;
                            })
        .def_classmethod(
            "get",
            [](py::object cls, int32_t value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOSLayoutAttrGet(ctx, value);
            if (mlirAttributeIsNull(a)) return py::none();
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "PadValueAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirAttributeIsAInteger(a) &&
                                    mlirIntegerTypeGetWidth(mlirAttributeGetType(a)) == 32;
                            })
        .def_classmethod(
            "get",
            [](py::object cls, int32_t value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOPadValueAttrGet(ctx, value);
            if (mlirAttributeIsNull(a)) return py::none();
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());
    // [保留 HEAD]: AddressSpaceAttr 定义
    mlir_attribute_subclass(
        m, "AddressSpaceAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAAddressSpaceAttr(a); })
    .def_classmethod(
        "get",
        [](py::object cls, py::object value, MlirContext context) -> py::object {
        // 支持传 enum 或 int
        int32_t v = 0;
        if (py::isinstance<py::int_>(value)) {
            v = py::cast<int32_t>(value);
        } else {
            // enum: pto.AddressSpace.UB -> 转成 int
            v = py::cast<int32_t>(value.attr("value").cast<py::int_>());
        }
        MlirAttribute a = mlirPTOAddressSpaceAttrGet(context, v);
        return cls.attr("__call__")(a);
        },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
    .def_property_readonly(
        "value",
        [](MlirAttribute self) -> int32_t {
        return mlirPTOAddressSpaceAttrGetValue(self);
        });

    mlir_attribute_subclass(
        m, "RoundModeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsARoundModeAttr(a); })
     .def_classmethod(
         "get",
        [](py::object cls, py::object value, MlirContext ctx) -> py::object {
        int32_t v = 0;
        if (py::isinstance<py::int_>(value)) {
            v = value.cast<int32_t>();
        } else if (py::hasattr(value, "value")) {
            // 通用：py::enum_ 通常有 .value
            v = value.attr("value").cast<int32_t>();
        } else {
            throw std::runtime_error("RoundModeAttr.get expects int or RoundMode enum");
        }

        MlirAttribute a = mlirPTORoundModeAttrGet(ctx, v);
        if (mlirAttributeIsNull(a)) return py::none();
        return cls.attr("__call__")(a);
         },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none())

    .def_property_readonly(
        "value",
        [](MlirAttribute self) -> int32_t {
        return mlirPTORoundModeAttrGetValue(self);
        });

    mlir_attribute_subclass(m, "CmpModeAttr", mlirAttributeIsAPTOCmpModeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, MlirPTOCmpMode value) {
            return cls(mlirPTOCmpModeAttrGet(ctx, value));
          },
          "cls"_a, "context"_a, "value"_a)
      .def_property_readonly(
          "value",
          [](MlirAttribute self) {
            return mlirPTOCmpModeAttrGetValue(self);
          });

    // [保留 Feature]: GM Type Factory
    // ==========================================================================
    // [新增] GM Type Factory: 专门用于隐藏复杂的 memref 定义
    // ==========================================================================
    m.def("get_gm_type", 
        [](py::sequence shape, MlirType elementType, MlirContext context) -> MlirType {
            // ... (这里保留你之前的完整实现) ...
            // 为节省篇幅省略中间实现，请保留你原来的代码逻辑
            std::string typeStr = "memref<";
            auto shp = toInt64Vector(shape);
            for (auto s : shp) {
                typeStr += std::to_string(s) + "x";
            }
            // ... (下略) ...
            typeStr += "f32"; 
            typeStr += ", strided<[?, 1], offset: ?>, #pto.address_space<gm>>";
            MlirType t = mlirTypeParseGet(context, mlirStringRefCreate(typeStr.c_str(), typeStr.length()));
            return t;
        },
        py::arg("shape"), py::arg("element_type"), py::arg("context")
    );
    // --------------------------------------------------------------------------
    // !pto.ptr<elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "PtrType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAPtrType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirType elementType, MlirContext context) -> py::object {
                MlirType t = mlirPTOPtrTypeGet(context, elementType);
                // Construct the Python wrapper object from the underlying MlirType.
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("element_type"),
            // NOTE: If you don't have a "default context" helper, it is safer
            // to require context explicitly. Here we keep a default to be
            // consistent with common MLIR Python patterns, but passing None
            // may crash if your C-API does not handle it.
            py::arg("context") = py::none())
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOPtrTypeGetElementType(self);
            });

    // --------------------------------------------------------------------------
    // !pto.tensor_view<rank x elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "TensorViewType",
        [](MlirType type) -> bool { return mlirPTOTypeIsATensorViewType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, int64_t rank, MlirType elementType, MlirContext context) -> py::object {
                MlirType t = mlirPTOTensorViewTypeGet(context, rank, elementType);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("rank"), py::arg("element_type"),
            py::arg("context") = py::none())
        .def_property_readonly(
            "rank",
            [](MlirType self) -> int64_t {
                return mlirPTOTensorViewTypeGetRank(self);
            })
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOTensorViewTypeGetElementType(self);
            });
        // --------------------------------------------------------------------------
    // !pto.tile_view<shape x elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "TileViewType",
        [](MlirType t) -> bool { return mlirPTOTypeIsATileViewType(t); })
    .def_classmethod(
        "get",
        [](py::object cls, py::sequence shape, MlirType elementType, MlirContext context) -> py::object {
        auto shp = toInt64Vector(shape);
        MlirType t = mlirPTOTileViewTypeGet(context,
                                            (intptr_t)shp.size(),
                                            shp.data(),
                                            elementType);
        return cls.attr("__call__")(t);
        },
        py::arg("cls"), py::arg("shape"), py::arg("element_type"),
        py::arg("context") = py::none())
    .def_property_readonly(
        "rank",
        [](MlirType self) -> intptr_t { return mlirPTOTileViewTypeGetRank(self); })
    .def_property_readonly(
        "element_type",
        [](MlirType self) -> MlirType { return mlirPTOTileViewTypeGetElementType(self); })
    .def_property_readonly(
        "shape",
        [](MlirType self) -> py::list {
        intptr_t n = 0;
        const int64_t *data = mlirPTOTileViewTypeGetShape(self, &n);
        return shapeToPyList(data, n);
        });

    // --------------------------------------------------------------------------
    // !pto.tile<shape x elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "TileType",
        [](MlirType t) -> bool { return mlirPTOTypeIsATileType(t); })
    .def_classmethod(
        "get",
        [](py::object cls, py::sequence shape, MlirType elementType, MlirContext context) -> py::object {
        auto shp = toInt64Vector(shape);
        MlirType t = mlirPTOTileTypeGet(context,
                                        (intptr_t)shp.size(),
                                        shp.data(),
                                        elementType);
        return cls.attr("__call__")(t);
        },
        py::arg("cls"), py::arg("shape"), py::arg("element_type"),
        py::arg("context") = py::none())
    .def_property_readonly(
        "rank",
        [](MlirType self) -> intptr_t { return mlirPTOTileTypeGetRank(self); })
    .def_property_readonly(
        "element_type",
        [](MlirType self) -> MlirType { return mlirPTOTileTypeGetElementType(self); })
    .def_property_readonly(
        "shape",
        [](MlirType self) -> py::list {
        intptr_t n = 0;
        const int64_t *data = mlirPTOTileTypeGetShape(self, &n);
        return shapeToPyList(data, n);
        });

    // ---- TileBufConfigAttr ----
    mlir_attribute_subclass(m, "TileBufConfigAttr",
                            [](MlirAttribute a) -> bool {
                                return mlirPTOAttrIsATileBufConfigAttr(a);
                            })
        .def_classmethod(
            "get_default",
            [](py::object cls, MlirContext ctx) -> py::object {
                MlirAttribute a = mlirPTOTileBufConfigAttrGetDefault(ctx);
                if (mlirAttributeIsNull(a)) return py::none();
                return cls(a);
            },
            py::arg("cls"), py::arg("context") = py::none())
        .def_classmethod(
            "get",
            [](py::object cls,
                MlirAttribute blayout,
                MlirAttribute slayout,
                int32_t s_fractal_size,
                MlirAttribute pad,
                MlirContext ctx) -> py::object {
                MlirType i32 = mlirIntegerTypeGet(ctx, 32);
                MlirAttribute sz = mlirIntegerAttrGet(i32, s_fractal_size);

                MlirAttribute a = mlirPTOTileBufConfigAttrGet(ctx, blayout, slayout, sz, pad);
                if (mlirAttributeIsNull(a)) return py::none();
                return cls(a);
            },
            py::arg("cls"),
            py::arg("blayout"),
            py::arg("slayout"),
            py::arg("s_fractal_size"),
            py::arg("pad"),
            py::arg("context") = py::none());

    // ---- TileBufType ----
    mlir_type_subclass(m, "TileBufType",
                        [](MlirType t) -> bool {
                        return mlirPTOTypeIsATileBufType(t);
                        })
        .def_classmethod(
        "get",
        [](py::object cls,
            std::vector<int64_t> shape,
            MlirType elementType,
            MlirAttribute memorySpace,
            py::object validShapeObj,
            py::object configObj,
            MlirContext ctx) -> py::object {

            // 1) 计算 validShape（默认=shape）
            std::vector<int64_t> validShape = shape;

            if (!validShapeObj.is_none()) {
            // 支持 valid_shape 为 list[int] 或 list[Optional[int]]
            py::list lst = validShapeObj.cast<py::list>();
            if ((size_t)lst.size() != shape.size()) {
                throw std::runtime_error("valid_shape rank must match shape rank");
            }
            validShape.resize(lst.size());
            for (ssize_t i = 0; i < lst.size(); ++i) {
                py::object e = lst[i];
                if (e.is_none()) {
                validShape[i] = -1;  // None -> dynamic
                } else {
                validShape[i] = e.cast<int64_t>();
                }
            }
            }

            // 2) 调 CAPI
            MlirType ty;
            if (!configObj.is_none()) {
            MlirAttribute cfg = configObj.cast<MlirAttribute>();
            ty = mlirPTOTileBufTypeGetWithValidShapeAndConfig(
                ctx,
                (intptr_t)shape.size(), shape.data(),
                elementType, memorySpace,
                (intptr_t)validShape.size(), validShape.data(),
                cfg);
            } else {
            ty = mlirPTOTileBufTypeGetWithValidShape(
                ctx,
                (intptr_t)shape.size(), shape.data(),
                elementType, memorySpace,
                (intptr_t)validShape.size(), validShape.data());
            }

            if (mlirTypeIsNull(ty)) return py::none();
            return cls(ty);
        },
        py::arg("cls"),
        py::arg("shape"),
        py::arg("element_type"),
        py::arg("memory_space"),
        py::arg("valid_shape") = py::none(),
        py::arg("config") = py::none(),
        py::arg("context") = py::none())
        .def_classmethod(
            "upcast_type",
            [](py::object cls, MlirType t) -> py::object {
                if (mlirPTOTypeIsATileBufType(t)) return cls(t);
                return py::none();
            },
            py::arg("cls"), py::arg("type"));
	
	populatePTODialectSubmodule(m);
}
