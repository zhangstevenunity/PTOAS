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
#include "mlir/IR/BuiltinTypes.h"
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
    .value("VEC",   mlir::pto::AddressSpace::VEC)
    .value("BIAS",   mlir::pto::AddressSpace::BIAS)
    .value("SCALING", mlir::pto::AddressSpace::SCALING)
    .export_values();
    py::enum_<mlir::pto::BLayout>(m, "BLayout")
    .value("RowMajor", mlir::pto::BLayout::RowMajor)
    .value("ColMajor", mlir::pto::BLayout::ColMajor);

    py::enum_<mlir::pto::SLayout>(m, "SLayout")
    .value("NoneBox", mlir::pto::SLayout::NoneBox)
    .value("RowMajor", mlir::pto::SLayout::RowMajor)
    .value("ColMajor", mlir::pto::SLayout::ColMajor);

    py::enum_<mlir::pto::PadValue>(m, "PadValue")
    .value("Null", mlir::pto::PadValue::Null)
    .value("Zero", mlir::pto::PadValue::Zero)
    .value("Max", mlir::pto::PadValue::Max)
    .value("Min", mlir::pto::PadValue::Min);

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

    py::enum_<mlir::pto::PIPE>(m, "PIPE")
      .value("PIPE_S", mlir::pto::PIPE::PIPE_S)
      .value("PIPE_V", mlir::pto::PIPE::PIPE_V)
      .value("PIPE_M", mlir::pto::PIPE::PIPE_M)
      .value("PIPE_MTE1", mlir::pto::PIPE::PIPE_MTE1)
      .value("PIPE_MTE2", mlir::pto::PIPE::PIPE_MTE2)
      .value("PIPE_MTE3", mlir::pto::PIPE::PIPE_MTE3)
      .value("PIPE_ALL", mlir::pto::PIPE::PIPE_ALL)
      .value("PIPE_MTE4", mlir::pto::PIPE::PIPE_MTE4)
      .value("PIPE_MTE5", mlir::pto::PIPE::PIPE_MTE5)
      .value("PIPE_V2", mlir::pto::PIPE::PIPE_V2)
      .value("PIPE_FIX", mlir::pto::PIPE::PIPE_FIX)
      .value("VIRTUAL_PIPE_MTE2_L1A", mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1A)
      .value("VIRTUAL_PIPE_MTE2_L1B", mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1B)
      .value("PIPE_NUM", mlir::pto::PIPE::PIPE_NUM)
      .value("PIPE_UNASSIGNED", mlir::pto::PIPE::PIPE_UNASSIGNED);

    py::enum_<mlir::pto::Layout>(m, "Layout")
      .value("ND", mlir::pto::Layout::ND)
      .value("DN", mlir::pto::Layout::DN)
      .value("NZ", mlir::pto::Layout::NZ);

    py::enum_<mlir::pto::SyncOpType>(m, "SyncOpType")
      .value("TLOAD", mlir::pto::SyncOpType::TLOAD)
      .value("TSTORE_ACC", mlir::pto::SyncOpType::TSTORE_ACC)
      .value("TSTORE_VEC", mlir::pto::SyncOpType::TSTORE_VEC)
      .value("TMOV_M2L", mlir::pto::SyncOpType::TMOV_M2L)
      .value("TMOV_M2S", mlir::pto::SyncOpType::TMOV_M2S)
      .value("TMOV_M2B", mlir::pto::SyncOpType::TMOV_M2B)
      .value("TMOV_M2V", mlir::pto::SyncOpType::TMOV_M2V)
      .value("TMOV_V2M", mlir::pto::SyncOpType::TMOV_V2M)
      .value("TMATMUL", mlir::pto::SyncOpType::TMATMUL)
      .value("TVEC", mlir::pto::SyncOpType::TVEC)
      .value("TVECWAIT_EVENT", mlir::pto::SyncOpType::TVECWAIT_EVENT)
      .export_values();

    py::enum_<mlir::pto::EVENT>(m, "EVENT")
      .value("EVENT_ID0", mlir::pto::EVENT::EVENT_ID0)
      .value("EVENT_ID1", mlir::pto::EVENT::EVENT_ID1)
      .value("EVENT_ID2", mlir::pto::EVENT::EVENT_ID2)
      .value("EVENT_ID3", mlir::pto::EVENT::EVENT_ID3)
      .value("EVENT_ID4", mlir::pto::EVENT::EVENT_ID4)
      .value("EVENT_ID5", mlir::pto::EVENT::EVENT_ID5)
      .value("EVENT_ID6", mlir::pto::EVENT::EVENT_ID6)
      .value("EVENT_ID7", mlir::pto::EVENT::EVENT_ID7)
      .export_values();

    py::enum_<mlir::pto::MaskPattern>(m, "MaskPattern")
      .value("P0101", mlir::pto::MaskPattern::P0101)
      .value("P0011", mlir::pto::MaskPattern::P0011)
      .value("P0110", mlir::pto::MaskPattern::P0110)
      .value("P0001", mlir::pto::MaskPattern::P0001)
      .value("P1111", mlir::pto::MaskPattern::P1111)
      .value("P1010", mlir::pto::MaskPattern::P1010)
      .export_values();

    mlir_attribute_subclass(m, "BLayoutAttr",
                        [](MlirAttribute a) -> bool {
                          return mlirPTOAttrIsABLayoutAttr(a);
                        })
    .def_classmethod(
        "get",
        [](py::object cls, mlir::pto::BLayout value, MlirContext ctx) -> py::object {
          MlirAttribute a = mlirPTOBLayoutAttrGet(ctx, static_cast<int32_t>(value));
          if (mlirAttributeIsNull(a)) return py::none();
          return cls(a);
        },
        py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "SLayoutAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsASLayoutAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, mlir::pto::SLayout value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOSLayoutAttrGet(ctx, static_cast<int32_t>(value));
            if (mlirAttributeIsNull(a)) return py::none();
            return cls(a);
            },
            py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

    mlir_attribute_subclass(m, "PadValueAttr",
                            [](MlirAttribute a) -> bool {
                            return mlirPTOAttrIsAPadValueAttr(a);
                            })
        .def_classmethod(
            "get",
            [](py::object cls, mlir::pto::PadValue value, MlirContext ctx) -> py::object {
            MlirAttribute a = mlirPTOPadValueAttrGet(ctx, static_cast<int32_t>(value));
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

    mlir_attribute_subclass(
        m, "PipeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAPipeAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = value.cast<int32_t>();
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("PipeAttr.get expects int or PIPE enum");
            }
            MlirAttribute a = mlirPTOPipeAttrGet(ctx, v);
            if (mlirAttributeIsNull(a))
              return py::none();
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOPipeAttrGetValue(self);
          });

    mlir_attribute_subclass(
        m, "LayoutAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsALayoutAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = value.cast<int32_t>();
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("LayoutAttr.get expects int or Layout enum");
            }
            MlirAttribute a = mlirPTOLayoutAttrGet(ctx, v);
            if (mlirAttributeIsNull(a))
              return py::none();
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOLayoutAttrGetValue(self);
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

    mlir_attribute_subclass(
        m, "SyncOpTypeAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsASyncOpTypeAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("SyncOpTypeAttr.get expects int or SyncOpType enum");
            }
            MlirAttribute a = mlirPTOSyncOpTypeAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) return py::none();
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOSyncOpTypeAttrGetValue(self);
          });

    mlir_attribute_subclass(
        m, "EventAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAEventAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("EventAttr.get expects int or EVENT enum");
            }
            MlirAttribute a = mlirPTOEventAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) return py::none();
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOEventAttrGetValue(self);
          });

    mlir_attribute_subclass(
        m, "MaskPatternAttr",
        [](MlirAttribute a) { return mlirPTOAttrIsAMaskPatternAttr(a); })
      .def_classmethod(
          "get",
          [](py::object cls, py::object value, MlirContext ctx) -> py::object {
            int32_t v = 0;
            if (py::isinstance<py::int_>(value)) {
              v = py::cast<int32_t>(value);
            } else if (py::hasattr(value, "value")) {
              v = value.attr("value").cast<int32_t>();
            } else {
              throw std::runtime_error("MaskPatternAttr.get expects int or MaskPattern enum");
            }
            MlirAttribute a = mlirPTOMaskPatternAttrGet(ctx, v);
            if (mlirAttributeIsNull(a)) return py::none();
            return cls.attr("__call__")(a);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> int32_t {
            return mlirPTOMaskPatternAttrGetValue(self);
          });

    // --------------------------------------------------------------------------
    // !pto.ptr<elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "PtrType",
        [](MlirType type) -> bool { return mlirPTOTypeIsAPtrType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, MlirType elementType,
               MlirContext context) -> py::object {
                MlirContext ctx = context;
                if (!ctx.ptr)
                    ctx = mlirTypeGetContext(elementType);
                MlirType t = mlirPTOPtrTypeGet(ctx, elementType);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("element_type"),
            py::arg("context") = py::none())
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOPtrTypeGetElementType(self);
            });

    // --------------------------------------------------------------------------
    // !pto.tensor_view<shape x elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "TensorViewType",
        [](MlirType type) -> bool { return mlirPTOTypeIsATensorViewType(type); })
        .def_classmethod(
            "get",
            [](py::object cls, py::object shape_or_rank, MlirType elementType, MlirContext context) -> py::object {
                std::vector<int64_t> shp;
                if (py::isinstance<py::int_>(shape_or_rank)) {
                    auto rank = shape_or_rank.cast<int64_t>();
                    shp.assign(static_cast<size_t>(rank), mlir::ShapedType::kDynamic);
                } else {
                    shp = toInt64Vector(shape_or_rank.cast<py::sequence>());
                }
                MlirType t = mlirPTOTensorViewTypeGet(
                    context, (intptr_t)shp.size(), shp.data(), elementType);
                return cls.attr("__call__")(t);
            },
            py::arg("cls"), py::arg("shape_or_rank"), py::arg("element_type"),
            py::arg("context") = py::none())
        .def_property_readonly(
            "rank",
            [](MlirType self) -> intptr_t { return mlirPTOTensorViewTypeGetRank(self); })
        .def_property_readonly(
            "element_type",
            [](MlirType self) -> MlirType {
                return mlirPTOTensorViewTypeGetElementType(self);
            })
        .def_property_readonly(
            "shape",
            [](MlirType self) -> py::list {
                intptr_t n = 0;
                const int64_t *data = mlirPTOTensorViewTypeGetShape(self, &n);
                return shapeToPyList(data, n);
            });
        // --------------------------------------------------------------------------
    // !pto.tile_view<shape x elem>
    // --------------------------------------------------------------------------
    mlir_type_subclass(
        m, "PartitionTensorViewType",
        [](MlirType t) -> bool { return mlirPTOTypeIsAPartitionTensorViewType(t); })
    .def_classmethod(
        "get",
        [](py::object cls, py::sequence shape, MlirType elementType, MlirContext context) -> py::object {
        auto shp = toInt64Vector(shape);
        MlirType t = mlirPTOPartitionTensorViewTypeGet(context,
                                            (intptr_t)shp.size(),
                                            shp.data(),
                                            elementType);
        return cls.attr("__call__")(t);
        },
        py::arg("cls"), py::arg("shape"), py::arg("element_type"),
        py::arg("context") = py::none())
    .def_property_readonly(
        "rank",
        [](MlirType self) -> intptr_t { return mlirPTOPartitionTensorViewTypeGetRank(self); })
    .def_property_readonly(
        "element_type",
        [](MlirType self) -> MlirType { return mlirPTOPartitionTensorViewTypeGetElementType(self); })
    .def_property_readonly(
        "shape",
        [](MlirType self) -> py::list {
        intptr_t n = 0;
        const int64_t *data = mlirPTOPartitionTensorViewTypeGetShape(self, &n);
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
