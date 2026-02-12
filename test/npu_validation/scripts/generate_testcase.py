#!/usr/bin/env python3
# coding=utf-8

import argparse
import ast
import re
from pathlib import Path
from typing import Optional

INCLUDE_REPLACEMENT = (
    "// ---------------------------------------------------------------------------\n"
    "// PTOAS compatibility layer\n"
    "//\n"
    "// The upstream pto-isa headers reference some FP8/FP4 types and the\n"
    "// __VEC_SCOPE__ marker that are not available on every AICore arch/toolchain\n"
    "// combination (e.g. __NPU_ARCH__==2201).\n"
    "//\n"
    "// For our PTOAS-generated kernels we don't rely on these types today, but the\n"
    "// headers still mention them in templates/static_asserts. Provide minimal\n"
    "// fallbacks to keep compilation working on dav-c220.\n"
    "// ---------------------------------------------------------------------------\n"
    "#ifndef __VEC_SCOPE__\n"
    "#define __VEC_SCOPE__\n"
    "#endif\n"
    "\n"
    "#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)\n"
    "typedef struct { unsigned char v; } hifloat8_t;\n"
    "typedef struct { unsigned char v; } float8_e4m3_t;\n"
    "typedef struct { unsigned char v; } float8_e5m2_t;\n"
    "typedef struct { unsigned char v; } float8_e8m0_t;\n"
    "typedef struct { unsigned char v; } float4_e1m2x2_t;\n"
    "typedef struct { unsigned char v; } float4_e2m1x2_t;\n"
    "#endif\n"
    "#include <stdint.h>\n"
    "\n"
    "// Some PTO-ISA types (e.g. TMRGSORT's MrgSortExecutedNumList) are defined\n"
    "// only in the CCE/AICore compilation path. The `bisheng -xcce` frontend\n"
    "// still performs a host-side pass that needs to parse kernel signatures.\n"
    "// Provide a minimal fallback so `launch.cpp` and host-side wrappers compile.\n"
    "#if !defined(__CCE_AICORE__)\n"
    "namespace pto {\n"
    "struct MrgSortExecutedNumList {\n"
    "    uint16_t mrgSortList0;\n"
    "    uint16_t mrgSortList1;\n"
    "    uint16_t mrgSortList2;\n"
    "    uint16_t mrgSortList3;\n"
    "};\n"
    "} // namespace pto\n"
    "#endif\n"
    "#include <pto/pto-inst.hpp>\n"
    "#include <pto/common/constants.hpp>\n"
    "#ifndef __CPU_SIM\n"
    "#include \"acl/acl.h\"\n"
    "#endif\n"
)


def _parse_shape(text: str):
    match = re.search(r"Shape<(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"Shape<\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 32, 32


def _is_gm_pointer_param(param: str) -> bool:
    return "__gm__" in param and "*" in param


def _extract_cpp_type(param: str) -> str:
    match = re.search(r"__gm__\s+([A-Za-z_]\w*)", param)
    if match:
        return match.group(1)

    tokens = param.replace("*", " ").strip().split()
    if not tokens:
        return "float"
    if len(tokens) == 1:
        return tokens[0]
    qualifiers = {"const", "volatile", "restrict", "__restrict", "__restrict__"}
    type_tokens = [t for t in tokens[:-1] if t not in qualifiers]
    return " ".join(type_tokens) if type_tokens else tokens[0]


def _extract_cpp_name(param: str) -> str:
    parts = param.strip().split()
    if not parts:
        return "arg"
    name = parts[-1].replace("*", "").strip()
    if name.startswith("__"):
        return "arg"
    return name


def _strip_param_name(raw: str, name: str) -> str:
    """
    Return the type part of a parameter declaration, keeping qualifiers and the
    pointer '*' but removing the trailing variable name.
    Example: "__gm__ float* v1" -> "__gm__ float*"
    """
    pattern = rf"\b{re.escape(name)}\b\s*$"
    stripped = re.sub(pattern, "", raw.strip())
    return stripped.strip()


def _infer_void_gm_pointee_type(text: str, param_name: str) -> Optional[str]:
    # Common patterns in PTOAS-generated kernels:
    #   __gm__ int16_t* v16 = (__gm__ int16_t*) v1;
    #   __gm__ half*   v16 = (__gm__ half*) v1;
    name = re.escape(param_name)
    patterns = [
        # Direct assignment after implicit conversion (some kernels keep the
        # ABI as `void*` and only materialize the real type for arithmetic).
        rf"__gm__\s+([A-Za-z_]\w*)\s*\*\s*\w+\s*=\s*{name}\b",
        rf"\(__gm__\s+([A-Za-z_]\w*)\s*\*\)\s*{name}\b",
        rf"reinterpret_cast<__gm__\s+([A-Za-z_]\w*)\s*\*\s*>\(\s*{name}\s*\)",
        rf"static_cast<__gm__\s+([A-Za-z_]\w*)\s*\*\s*>\(\s*{name}\s*\)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            ty = match.group(1)
            if ty and ty != "void":
                return ty
    return None


def _detect_output_pointer_param(text: str, pointer_param_names):
    if not pointer_param_names:
        return None

    tstore_gts = re.findall(r"\bTSTORE\s*\(\s*(\w+)\s*,", text)
    if not tstore_gts:
        return None

    gt_to_ptr = {}
    for m in re.finditer(r"\b(\w+)\s*=\s*[\w:<>]+\s*\(\s*(\w+)\s*[,)]", text):
        gt_to_ptr[m.group(1)] = m.group(2)

    ptr_to_base = {}
    for m in re.finditer(r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*\+", text):
        ptr_to_base[m.group(1)] = m.group(2)

    ptr_to_param = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b",
        text,
    ):
        ptr_to_param[m.group(1)] = m.group(2)

    def resolve_param(ptr: Optional[str]) -> Optional[str]:
        if not ptr:
            return None
        cur = ptr
        seen = set()
        for _ in range(8):
            if cur in seen:
                break
            seen.add(cur)
            if cur in pointer_param_names:
                return cur
            mapped = ptr_to_param.get(cur)
            if mapped in pointer_param_names:
                return mapped
            cur = ptr_to_base.get(cur)
            if cur is None:
                break
        return None

    for gt in tstore_gts:
        ptr = gt_to_ptr.get(gt)
        if not ptr:
            continue
        resolved = resolve_param(ptr)
        if resolved:
            return resolved
    return None


def _parse_kernel_params(text: str):
    match = re.search(r"__global__\s+(?:\w+\s+)*void\s+\w+\s*\(([^)]*)\)", text, re.S)
    if not match:
        return []
    params_blob = match.group(1).strip()
    if not params_blob:
        return []
    params = []
    depth = 0
    start = 0
    for idx, ch in enumerate(params_blob):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            params.append(params_blob[start:idx].strip())
            start = idx + 1
    last = params_blob[start:].strip()
    if last:
        params.append(last)
    return params


def _parse_kernel_name(text: str) -> str:
    match = re.search(r"__global__\s+(?:\w+\s+)*void\s+(\w+)\s*\(", text, re.S)
    return match.group(1) if match else "kernel"


def _np_dtype_for_cpp(cpp_type: str) -> str:
    mapping = {
        "float": "np.float32",
        "half": "np.float16",
        "aclFloat16": "np.float16",
        "__bf16": "np.uint16",
        "bfloat16_t": "np.uint16",
        "int8_t": "np.int8",
        "uint8_t": "np.uint8",
        "int16_t": "np.int16",
        "uint16_t": "np.uint16",
        "int32_t": "np.int32",
        "uint32_t": "np.uint32",
        "int64_t": "np.int64",
        "uint64_t": "np.uint64",
    }
    return mapping.get(cpp_type, "np.float32")


def _cpp_host_type(cpp_type: str) -> str:
    if cpp_type == "half":
        return "aclFloat16"
    if cpp_type in {"__bf16", "bfloat16_t"}:
        return "uint16_t"
    return cpp_type


def _default_eps_for_cpp_type(cpp_type: str) -> float:
    # CPU golden vs NPU results may have small floating-point differences.
    if cpp_type in {"half", "aclFloat16"}:
        return 1e-2
    if cpp_type in {"float"}:
        return 1e-4
    return 0.0


def _derive_testcase_name(input_cpp: Path) -> str:
    name = input_cpp.stem
    if name.endswith("-pto"):
        name = name[:-4]
    if name.endswith("_pto"):
        name = name[:-4]
    return name


def _replace_includes(text: str) -> str:
    if "#include \"common/pto_instr.hpp\"" in text:
        return text.replace("#include \"common/pto_instr.hpp\"", INCLUDE_REPLACEMENT.rstrip())
    if "#include <pto/pto-inst.hpp>" in text:
        return text
    return INCLUDE_REPLACEMENT + "\n" + text


def _inject_packed_pred_mask_preload(
    kernel_text: str,
    *,
    dst_tile: str,
    output_ptr: str,
    output_cpp_type: str,
    rows: int,
    cols: int,
    logical_elem_count: int,
) -> str:
    """
    pto.tcmp / pto.tcmps write a packed predicate mask and may leave parts of the
    destination tile undefined (UB garbage). Our validation harness compares
    two NPU runs for determinism; undefined bytes make the compare flaky.

    Inject a TLOAD(dst, GM_output) before the first PIPE_MTE2->PIPE_V barrier so
    the whole dst tile starts from deterministic contents (the output buffer is
    initialized from .bin files on the host).
    """
    if "PTOAS_PACKED_MASK_PRELOAD" in kernel_text:
        return kernel_text

    if not dst_tile or not output_ptr:
        return kernel_text

    # Find a reasonable insertion point: before the first MTE2->V set_flag.
    m = re.search(r"^(\s*)set_flag\s*\(\s*PIPE_MTE2\s*,\s*PIPE_V\s*,", kernel_text, re.M)
    if m:
        indent = m.group(1)
        insert_at = m.start()
    else:
        # Fallback: insert right before the first TCMP/TCMPS call.
        m2 = re.search(r"^(\s*)TCMPS?\s*\(", kernel_text, re.M)
        if not m2:
            return kernel_text
        indent = m2.group(1)
        insert_at = m2.start()

    # We don't rely on the kernel's existing GlobalTensor aliases here; keep
    # names unique to avoid collisions.
    preload_lines = [
        f"{indent}// PTOAS_PACKED_MASK_PRELOAD: init packed predicate dst from GM",
        f"{indent}{{",
        f"{indent}  using __ptoas_mask_gt_shape = pto::Shape<1, 1, 1, {rows}, {cols}>;",
        f"{indent}  using __ptoas_mask_gt_stride = pto::Stride<{logical_elem_count}, {logical_elem_count}, {logical_elem_count}, {cols}, 1>;",
        f"{indent}  constexpr pto::Layout __ptoas_mask_gt_layout = pto::Layout::ND;",
        f"{indent}  __ptoas_mask_gt_shape __ptoas_mask_shape = __ptoas_mask_gt_shape();",
        f"{indent}  __ptoas_mask_gt_stride __ptoas_mask_stride = __ptoas_mask_gt_stride();",
        f"{indent}  using __ptoas_mask_gt = GlobalTensor<{output_cpp_type}, __ptoas_mask_gt_shape, __ptoas_mask_gt_stride, __ptoas_mask_gt_layout>;",
        f"{indent}  __ptoas_mask_gt __ptoas_mask_src = __ptoas_mask_gt((__gm__ {output_cpp_type}*){output_ptr}, __ptoas_mask_shape, __ptoas_mask_stride);",
        f"{indent}  TLOAD({dst_tile}, __ptoas_mask_src);",
        f"{indent}}}",
        "",
    ]
    block = "\n".join(preload_lines)
    return kernel_text[:insert_at] + block + kernel_text[insert_at:]


def _infer_aicore_arch(kernel_text: str, soc_version: str) -> str:
    # Heuristic: kernels that touch cube/L0/L1 tile types or cbuf memories need
    # the "cube" arch; pure vector kernels can use the vector arch.
    #
    # IMPORTANT: the default arch depends on the Ascend SoC.
    cube_markers = (
        "TileType::Mat",
        "TileType::Left",
        "TileType::Right",
        "TileType::Acc",
        "__cbuf__",
        "__ca__",
        "__cb__",
        "__cc__",
        "copy_gm_to_cbuf",
        "copy_cbuf_to_gm",
        "mad(",
        "mmad(",
        "TMMAD",
    )
    needs_cube = any(m in kernel_text for m in cube_markers)

    sv = (soc_version or "").lower()
    if "910b" in sv:
        # Ascend910B* (e.g. Ascend910B1) uses dav-c310 toolchain arch.
        return "dav-c310-cube" if needs_cube else "dav-c310-vec"

    # Default to Ascend910 (dav-c220) when SoC is unknown.
    return "dav-c220-cube" if needs_cube else "dav-c220-vec"


def _parse_int_list(blob: str):
    items = []
    for part in blob.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p, 0))
        except ValueError:
            return None
    return items


def _infer_mrgsort_block_len(kernel_text: str) -> Optional[int]:
    """
    Try to infer the compile-time blockLen argument passed to:
        TMRGSORT(dst, src, blockLen)

    Most PTOAS-generated kernels use a constant like:
        int32_t v3 = 64;
        TMRGSORT(v22, v21, v3);
    """
    call = re.search(r"\bTMRGSORT\s*\(\s*\w+\s*,\s*\w+\s*,\s*([^)]+?)\s*\)", kernel_text)
    if not call:
        return None
    arg = call.group(1).strip()
    # Direct literal.
    if re.fullmatch(r"(?:0x[0-9A-Fa-f]+|\d+)", arg):
        try:
            return int(arg, 0)
        except ValueError:
            return None

    # Identifier that is defined as a constant earlier in the kernel.
    if not re.fullmatch(r"[A-Za-z_]\w*", arg):
        return None
    match = re.search(rf"\b(?:int32_t|uint32_t|int|unsigned)\s+{re.escape(arg)}\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;", kernel_text)
    if not match:
        return None
    try:
        return int(match.group(1), 0)
    except ValueError:
        return None


def _required_elements_for_shape_stride(shape_dims, stride_dims) -> Optional[int]:
    if not shape_dims or not stride_dims:
        return None
    n = min(len(shape_dims), len(stride_dims))
    req = 1
    for i in range(n):
        dim = shape_dims[i]
        stride = stride_dims[i]
        if not isinstance(dim, int) or not isinstance(stride, int):
            return None
        if dim <= 0:
            continue
        req += (dim - 1) * stride
    return max(req, 1)


def _sanitize_int_expr(expr: str) -> str:
    expr = expr.strip()
    # Strip common C-style integer casts found in PTOAS-generated code.
    expr = re.sub(
        r"\(\s*(?:unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s*\)",
        "",
        expr,
    )
    # Strip integer literal suffixes (u/l/ul/ull...).
    expr = re.sub(r"(\b0x[0-9A-Fa-f]+|\b\d+)(?:[uUlL]+)\b", r"\1", expr)
    return expr.strip()


def _safe_eval_int_expr(expr: str, env: dict) -> Optional[int]:
    """
    Best-effort evaluate a C-like integer expression using values from `env`.

    Returns None if the expression contains unknown identifiers or unsupported
    constructs.
    """
    expr = _sanitize_int_expr(expr)
    if not expr:
        return None

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, bool)):
                return int(node.value)
            return None
        if isinstance(node, ast.Name):
            if node.id in env and env[node.id] is not None:
                return int(env[node.id])
            return None
        if isinstance(node, ast.UnaryOp):
            val = ev(node.operand)
            if val is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            return None
        if isinstance(node, ast.BinOp):
            left = ev(node.left)
            right = ev(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right if right != 0 else None
            if isinstance(node.op, ast.Mod):
                return left % right if right != 0 else None
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.RShift):
                return left >> right
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitOr):
                return left | right
            if isinstance(node.op, ast.BitXor):
                return left ^ right
            return None
        return None

    return ev(parsed)


def _infer_int_var_maxima(kernel_text: str) -> dict:
    """
    Infer max values for simple integer temporaries (e.g. v23) used in pointer
    arithmetic, by evaluating constant-ish assignments and simple for-loop ranges.

    This is used to size GM buffers conservatively for CPU/NPU runs.
    """
    assigns = []

    int_vars = set()
    for m in re.finditer(
        r"\b(?:bool|unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s+(\w+)\s*(?:=\s*[^;]+)?;",
        kernel_text,
    ):
        int_vars.add(m.group(1))

    # Typed initialization (non-hoisted case).
    for m in re.finditer(
        r"\b(?:bool|unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s+(\w+)\s*=\s*([^;]+);",
        kernel_text,
    ):
        name = m.group(1)
        expr = m.group(2).strip()
        assigns.append((name, expr))

    # declareVariablesAtTop hoists declarations, leaving untyped assignments like:
    #   v34 = v29 + v33;
    for m in re.finditer(r"\b(\w+)\s*=\s*([^;]+);", kernel_text):
        name = m.group(1)
        if name not in int_vars:
            continue
        expr = m.group(2).strip()
        assigns.append((name, expr))

    loops = []
    for m in re.finditer(
        r"for\s*\(\s*(?:unsigned|int|long|size_t|int(?:8|16|32|64)_t|uint(?:8|16|32|64)_t)\s+(\w+)\s*=\s*([^;]+?)\s*;\s*\1\s*<\s*([^;]+?)\s*;\s*\1\s*\+=\s*([^)]+?)\s*\)",
        kernel_text,
    ):
        ind = m.group(1)
        start = m.group(2).strip()
        end = m.group(3).strip()
        step = m.group(4).strip()
        loops.append((ind, start, end, step))

    maxima: dict[str, Optional[int]] = {}

    def set_max(name: str, value: int) -> bool:
        cur = maxima.get(name)
        if cur is None or value > cur:
            maxima[name] = value
            return True
        return False

    changed = True
    for _ in range(64):
        if not changed:
            break
        changed = False
        for name, expr in assigns:
            val = _safe_eval_int_expr(expr, maxima)
            if val is None:
                continue
            if set_max(name, val):
                changed = True

        for ind, start, end, step in loops:
            start_v = _safe_eval_int_expr(start, maxima)
            end_v = _safe_eval_int_expr(end, maxima)
            step_v = _safe_eval_int_expr(step, maxima)
            if start_v is None or end_v is None or step_v is None:
                continue
            if step_v == 0:
                continue
            if step_v > 0:
                if end_v <= start_v:
                    max_ind = start_v
                else:
                    span = end_v - start_v - 1
                    max_ind = start_v + (span // step_v) * step_v
            else:
                # Rare in these kernels; approximate with start.
                max_ind = start_v
            if set_max(ind, max_ind):
                changed = True

    # Replace None with 0 for downstream best-effort arithmetic.
    return {k: (0 if v is None else int(v)) for k, v in maxima.items()}


def _infer_gm_pointer_elem_counts(kernel_text: str, pointer_param_names):
    """
    Infer minimum element counts for each __gm__ pointer param from GlobalTensor
    shape/stride metadata found in PTOAS-generated kernels.

    This fixes cases where the logical shape is small (e.g. 32x32) but the GM
    tensor uses padded strides (e.g. row stride 256), so the kernel accesses a
    much larger linear range.
    """
    if not pointer_param_names:
        return {}

    pointer_params = set(pointer_param_names)

    int_max = _infer_int_var_maxima(kernel_text)

    pointer_like = set(pointer_param_names)
    for m in re.finditer(r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*(?:=[^;]+)?;", kernel_text):
        pointer_like.add(m.group(1))

    ptr_to_base_offset = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*\+\s*([^;]+);",
        kernel_text,
    ):
        ptr_to_base_offset[m.group(1)] = (m.group(2), m.group(3).strip())

    # declareVariablesAtTop form:
    #   __gm__ float* v35;
    #   v35 = v1 + v34;
    for m in re.finditer(r"\b(\w+)\s*=\s*(\w+)\s*\+\s*([^;]+);", kernel_text):
        lhs = m.group(1)
        base = m.group(2)
        if lhs not in pointer_like:
            continue
        if base not in pointer_like and base not in pointer_params:
            continue
        ptr_to_base_offset[lhs] = (base, m.group(3).strip())

    ptr_to_param = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b",
        kernel_text,
    ):
        ptr_to_param[m.group(1)] = m.group(2)

    for m in re.finditer(r"\b(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b", kernel_text):
        lhs = m.group(1)
        rhs = m.group(2)
        if lhs not in pointer_like:
            continue
        if rhs not in pointer_like and rhs not in pointer_params:
            continue
        ptr_to_param[lhs] = rhs

    def resolve_param_and_offset(ptr: str):
        cur = ptr
        offset = 0
        seen = set()
        for _ in range(16):
            if cur in pointer_params:
                return cur, offset
            if cur in seen:
                break
            seen.add(cur)
            mapped = ptr_to_param.get(cur)
            if mapped:
                cur = mapped
                continue
            base_off = ptr_to_base_offset.get(cur)
            if base_off:
                base, off_expr = base_off
                off_val = _safe_eval_int_expr(off_expr, int_max)
                if off_val is not None:
                    offset += max(off_val, 0)
                cur = base
                continue
            break
        return None, None

    # Parse aliases: GTShape_*=pto::Shape<...>; GTStride_*=pto::Stride<...>;
    shape_aliases = {}
    for m in re.finditer(r"using\s+(\w+)\s*=\s*pto::Shape<([^>]*)>;", kernel_text):
        dims = _parse_int_list(m.group(2))
        if dims:
            shape_aliases[m.group(1)] = dims

    stride_aliases = {}
    for m in re.finditer(r"using\s+(\w+)\s*=\s*pto::Stride<([^>]*)>;", kernel_text):
        dims = _parse_int_list(m.group(2))
        if dims:
            stride_aliases[m.group(1)] = dims

    # Map GT_* alias -> (shape_alias, stride_alias)
    gt_alias_to_shape_stride = {}
    for m in re.finditer(
        # Matches both:
        #   using GT = GlobalTensor<T, ShapeAlias, StrideAlias>;
        # and the 4-param layout form:
        #   using GT = GlobalTensor<T, ShapeAlias, StrideAlias, LayoutAlias>;
        r"using\s+(\w+)\s*=\s*GlobalTensor<\s*[^,>]+\s*,\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*[^>]+)?\s*>;",
        kernel_text,
    ):
        gt_alias = m.group(1)
        shape_alias = m.group(2)
        stride_alias = m.group(3)
        gt_alias_to_shape_stride[gt_alias] = (shape_alias, stride_alias)

    # Find instantiations: GT_xxx v = GT_xxx(ptr, ...)
    param_elem_counts = {}
    for m in re.finditer(r"\b(\w+)\s+\w+\s*=\s*\1\s*\(\s*(\w+)\s*,", kernel_text):
        gt_alias = m.group(1)
        base_ptr = m.group(2)
        shape_stride = gt_alias_to_shape_stride.get(gt_alias)
        if not shape_stride:
            continue
        shape_dims = shape_aliases.get(shape_stride[0])
        stride_dims = stride_aliases.get(shape_stride[1])
        req = _required_elements_for_shape_stride(shape_dims, stride_dims)
        if not req:
            continue
        param, off = resolve_param_and_offset(base_ptr)
        if not param or off is None:
            continue
        param_elem_counts[param] = max(param_elem_counts.get(param, 0), req + max(off, 0))

    # Newer PTOAS EmitC output (especially with declareVariablesAtTop) may avoid
    # `using GTShape = ...; using GTStride = ...;` aliases and instead embeds
    # pto::Shape/pto::Stride directly in the GlobalTensor template.
    for m in re.finditer(
        r"\b(?:pto::)?GlobalTensor<[^;\n]*(?:pto::)?Shape<([^>]*)>[^;\n]*(?:pto::)?Stride<([^>]*)>[^;\n]*>\s*\(\s*(\w+)\s*,",
        kernel_text,
    ):
        shape_dims = _parse_int_list(m.group(1))
        stride_dims = _parse_int_list(m.group(2))
        req = _required_elements_for_shape_stride(shape_dims, stride_dims)
        if not req:
            continue
        base_ptr = m.group(3)
        param, off = resolve_param_and_offset(base_ptr)
        if not param or off is None:
            continue
        param_elem_counts[param] = max(param_elem_counts.get(param, 0), req + max(off, 0))

    return param_elem_counts


def generate_testcase(
    input_cpp: Path,
    output_root: Optional[Path],
    testcase: str,
    run_mode: str,
    soc_version: str,
    aicore_arch: Optional[str] = None,
):
    sample_dir = input_cpp.parent
    if output_root:
        output_dir = output_root / sample_dir.name / testcase
    else:
        output_dir = sample_dir / "npu_validation" / testcase
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_kernel = input_cpp.read_text(encoding="utf-8")
    raw_kernel_for_analysis = raw_kernel
    # pto.tcmp / pto.tcmps produce packed predicate masks and leave parts of the
    # logical u8 tile undefined. This can make byte-wise compares flaky.
    has_packed_pred_mask = re.search(r"\bTCMPS?\s*\(", raw_kernel_for_analysis) is not None
    has_dav_cube = "__DAV_CUBE__" in raw_kernel
    has_dav_vec = "__DAV_VEC__" in raw_kernel

    if aicore_arch is None:
        # Sectioned kernels contain `#if defined(__DAV_CUBE__)` / `__DAV_VEC__`
        # blocks. They frequently rely on cross-section synchronization (e.g.
        # set_flag in cube section + wait_flag in vector section). If we build
        # with a cube-only arch, common vector intrinsics (vabs/set_vector_mask)
        # may be unavailable; build with a vector arch and explicitly enable the
        # section macros instead.
        if has_dav_cube or has_dav_vec:
            sv = (soc_version or "").lower()
            aicore_arch = "dav-c310-vec" if "910b" in sv else "dav-c220-vec"
        else:
            aicore_arch = _infer_aicore_arch(raw_kernel, soc_version)

    # Force-define DAV section macros so both sections are compiled into the
    # same binary. This keeps the generated validation executable self-contained
    # and avoids deadlocks when one side of a set/wait pair is compiled out.
    dav_defines = ""
    if has_dav_cube:
        dav_defines += " -D__DAV_CUBE__"
    if has_dav_vec:
        dav_defines += " -D__DAV_VEC__"

    rows, cols = _parse_shape(raw_kernel_for_analysis)
    logical_elem_count = rows * cols
    kernel_name = _parse_kernel_name(raw_kernel_for_analysis)
    raw_params = _parse_kernel_params(raw_kernel_for_analysis)
    mrgsort_block_len = _infer_mrgsort_block_len(raw_kernel_for_analysis) if "TMRGSORT" in raw_kernel_for_analysis else None

    pointer_param_names = [_extract_cpp_name(p) for p in raw_params if _is_gm_pointer_param(p)]
    inferred_void_ptr_types = {}
    for raw in raw_params:
        if not _is_gm_pointer_param(raw):
            continue
        name = _extract_cpp_name(raw)
        cpp_type = _extract_cpp_type(raw)
        if cpp_type == "void":
            inferred = _infer_void_gm_pointee_type(raw_kernel_for_analysis, name)
            if inferred:
                inferred_void_ptr_types[name] = inferred

    output_ptr = _detect_output_pointer_param(raw_kernel_for_analysis, pointer_param_names)
    if output_ptr is None and pointer_param_names:
        output_ptr = pointer_param_names[0] if len(pointer_param_names) == 1 else pointer_param_names[-1]

    params = []
    for raw in raw_params:
        name = _extract_cpp_name(raw)
        cpp_type = _extract_cpp_type(raw)
        if cpp_type == "void" and name in inferred_void_ptr_types:
            cpp_type = inferred_void_ptr_types[name]
        if _is_gm_pointer_param(raw):
            params.append(
                {
                    "kind": "ptr",
                    "raw": raw,
                    "name": name,
                    "cpp_type": cpp_type,
                    "host_type": _cpp_host_type(cpp_type),
                    "role": "output" if name == output_ptr else "input",
                }
            )
        else:
            params.append(
                {
                    "kind": "scalar",
                    "raw": raw,
                    "name": name,
                    "cpp_type": cpp_type,
                    "host_type": _cpp_host_type(cpp_type),
                }
            )

    # Initialize every GM pointer from a host-side .bin file.
    #
    # Rationale:
    # - Some kernels are in-place (single pointer param) or may read from an
    #   "output" pointer as scratch. Leaving buffers uninitialized leads to
    #   non-determinism between CPU golden and real NPU.
    init_ptrs = [p for p in params if p["kind"] == "ptr"]
    output_ptrs = [p for p in params if p["kind"] == "ptr" and p["role"] == "output"]

    ptr_elem_counts = {p["name"]: logical_elem_count for p in params if p["kind"] == "ptr"}
    inferred_counts = _infer_gm_pointer_elem_counts(raw_kernel_for_analysis, pointer_param_names)
    for name, cnt in inferred_counts.items():
        ptr_elem_counts[name] = max(ptr_elem_counts.get(name, logical_elem_count), cnt)

    templates_root = Path(__file__).resolve().parents[1] / "templates"
    template = (templates_root / "main_template.cpp").read_text(encoding="utf-8")
    case_name = f"case_{rows}x{cols}"

    launch_name = f"Launch{kernel_name[0].upper()}{kernel_name[1:]}"

    launch_decl_params = []
    launch_call_args = []
    for p in params:
        if p["kind"] == "ptr":
            launch_decl_params.append(f"{p['host_type']} *{p['name']}")
            launch_call_args.append(f"{p['name']}Device")
        else:
            launch_decl_params.append(f"{p['host_type']} {p['name']}")
            launch_call_args.append(p["name"])

    param_decls_lines = []
    if any(p["kind"] == "ptr" for p in params):
        for p in params:
            if p["kind"] != "ptr":
                continue
            elem_cnt = ptr_elem_counts.get(p["name"], logical_elem_count)
            param_decls_lines.append(f"    size_t elemCount_{p['name']} = {elem_cnt};")
            param_decls_lines.append(
                f"    size_t fileSize_{p['name']} = elemCount_{p['name']} * sizeof({p['host_type']});"
            )

    for p in params:
        if p["kind"] != "scalar":
            continue
        t = p["host_type"]
        # Some PTO-ISA APIs use small POD structs as scalar parameters.
        # Example: pto::MrgSortExecutedNumList (used by TMRGSORT multi-list variants).
        if t.endswith("MrgSortExecutedNumList"):
            # A zero-initialized executed list can lead to illegal configurations
            # and runtime exceptions for TMRGSORT format2 on NPU. Default to "all
            # lists full" for our generated samples (each list holds 128 packed
            # structures in the standard 1x256 f32 representation).
            param_decls_lines.append(f"    {t} {p['name']}{{128, 128, 128, 128}};")
            continue
        if t == "bool":
            value = "true"
        elif re.match(r"^(u?int)(8|16|32|64)_t$", t) or t in {"int", "unsigned", "size_t"}:
            value = "1"
        elif t in {"float"}:
            value = "1.0f"
        elif t in {"double"}:
            value = "1.0"
        else:
            value = "0"
        param_decls_lines.append(f"    {t} {p['name']} = {value};")

    for p in params:
        if p["kind"] != "ptr":
            continue
        param_decls_lines.append(f"    {p['host_type']} *{p['name']}Host = nullptr;")
        param_decls_lines.append(f"    {p['host_type']} *{p['name']}Device = nullptr;")

    alloc_host = []
    alloc_device = []
    free_host = []
    free_device = []
    for p in params:
        if p["kind"] != "ptr":
            continue
        size_var = f"fileSize_{p['name']}"
        alloc_host.append(
            f"    ACL_CHECK(aclrtMallocHost((void **)(&{p['name']}Host), {size_var}));"
        )
        alloc_device.append(
            f"    ACL_CHECK(aclrtMalloc((void **)&{p['name']}Device, {size_var}, ACL_MEM_MALLOC_HUGE_FIRST));"
        )
        free_device.append(f"    aclrtFree({p['name']}Device);")
        free_host.append(f"    aclrtFreeHost({p['name']}Host);")

    read_inputs = []
    copy_inputs = []
    for p in init_ptrs:
        size_var = f"fileSize_{p['name']}"
        read_inputs.append(
            f"    ReadFile(\"./{p['name']}.bin\", {size_var}, {p['name']}Host, {size_var});"
        )
        copy_inputs.append(
            f"    ACL_CHECK(aclrtMemcpy({p['name']}Device, {size_var}, {p['name']}Host, {size_var}, ACL_MEMCPY_HOST_TO_DEVICE));"
        )

    output_copy_back = []
    output_write = []
    for p in output_ptrs:
        size_var = f"fileSize_{p['name']}"
        output_copy_back.append(
            f"    ACL_CHECK(aclrtMemcpy({p['name']}Host, {size_var}, {p['name']}Device, {size_var}, ACL_MEMCPY_DEVICE_TO_HOST));"
        )
        output_write.append(
            f"    WriteFile(\"./{p['name']}.bin\", {p['name']}Host, {size_var});"
        )

    param_decls = "\n".join(param_decls_lines)
    main_cpp = (
        template
        .replace("@TEST_SUITE@", testcase.upper())
        .replace("@CASE_NAME@", case_name)
        .replace(
            "@LAUNCH_DECL@",
            f"void {launch_name}({', '.join(launch_decl_params + ['void *stream'])});",
        )
        .replace("@PARAM_DECLS@", param_decls)
        .replace("@ALLOC_HOST@", "\n".join(alloc_host))
        .replace("@ALLOC_DEVICE@", "\n".join(alloc_device))
        .replace("@READ_INPUTS@", "\n".join(read_inputs))
        .replace("@COPY_TO_DEVICE@", "\n".join(copy_inputs))
        .replace(
            "@LAUNCH_CALL@",
            f"    {launch_name}({', '.join(launch_call_args + ['stream'])});",
        )
        .replace("@COPY_BACK@", "\n".join(output_copy_back))
        .replace("@WRITE_OUTPUT@", "\n".join(output_write))
        .replace("@FREE_DEVICE@", "\n".join(free_device))
        .replace("@FREE_HOST@", "\n".join(free_host))
    )
    (output_dir / "main.cpp").write_text(main_cpp, encoding="utf-8")

    golden_template = (templates_root / "golden_template.py").read_text(encoding="utf-8")
    input_generate = []
    elem_count = logical_elem_count
    # Some kernels use an integer tensor as "indices". The safe in-range domain
    # depends on the op semantics. For the pto-isa a2a3 implementations:
    # - TSCATTER: indices are linear indices in [0, rows*cols)
    # - TGATHER/TGATHERB: indices are linear indices in [0, rows*cols)
    index_mod = None
    if "TSCATTER" in raw_kernel:
        index_mod = max(elem_count, 1)
    elif any(m in raw_kernel for m in ("TGATHER", "TGATHERB")):
        index_mod = max(elem_count, 1)
    mrgsort_packed = "TMRGSORT" in raw_kernel
    for p in init_ptrs:
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        name = p["name"]
        size = ptr_elem_counts.get(name, elem_count)
        is_output = p.get("role") == "output"
        # If the kernel has both inputs and outputs, default to zero-init for
        # output buffers to match pto-isa ST conventions (and improve determinism).
        zero_init = is_output and len(init_ptrs) > 1

        if zero_init:
            input_generate.append(f"    {name} = np.zeros(({size},), dtype={np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        elif mrgsort_packed and (not is_output) and np_dtype in ("np.float32", "np.float16"):
            input_generate.append(f"    # TMRGSORT expects packed (value, index) structures (8 bytes each).")
            input_generate.append(f"    # Generate per-block sorted inputs to match pto-isa ST data layout.")
            if np_dtype == "np.float32":
                input_generate.append(f"    {name}__words_per_struct = 2  # float32(4B) + uint32(4B)")
                input_generate.append(f"    {name}__struct_dtype = np.dtype([('v', np.float32), ('i', np.uint32)])")
                input_generate.append(f"    {name}__value_dtype = np.float32")
            else:
                input_generate.append(f"    {name}__words_per_struct = 4  # float16(2B) + pad(2B) + uint32(4B)")
                input_generate.append(
                    f"    {name}__struct_dtype = np.dtype([('v', np.float16), ('pad', np.uint16), ('i', np.uint32)])"
                )
                input_generate.append(f"    {name}__value_dtype = np.float16")

            input_generate.append(f"    {name}__struct_count = {size} // {name}__words_per_struct")
            # Two modes:
            #   - Single-list format (TMRGSORT(dst, src, blockLen)): input is arranged in
            #     4 blocks and each block is sorted independently.
            #   - Multi-list format (TMRGSORT(dst, executed, tmp, src0..)): each input list
            #     is fully sorted.
            mrgsort_single = mrgsort_block_len is not None
            if mrgsort_single:
                input_generate.append(f"    {name}__block_len = {mrgsort_block_len}")
                input_generate.append(f"    {name}__structs_per_block = {name}__block_len // {name}__words_per_struct")
            input_generate.append(
                f"    {name}__values = np.random.uniform(low=0, high=1, size=({name}__struct_count,)).astype({name}__value_dtype)"
            )
            input_generate.append(f"    {name}__idx = np.arange({name}__struct_count, dtype=np.uint32)")
            if mrgsort_single:
                input_generate.append(f"    if {name}__structs_per_block > 0 and {name}__struct_count > 0:")
                input_generate.append(f"        pad = (-{name}__struct_count) % {name}__structs_per_block")
                input_generate.append(f"        if pad:")
                input_generate.append(
                    f"            {name}__values = np.concatenate(({name}__values, np.zeros(pad, dtype={name}__values.dtype)))"
                )
                input_generate.append(
                    f"            {name}__idx = np.concatenate(({name}__idx, np.zeros(pad, dtype={name}__idx.dtype)))"
                )
                input_generate.append(f"        v = {name}__values.reshape(-1, {name}__structs_per_block)")
                input_generate.append(f"        i = {name}__idx.reshape(-1, {name}__structs_per_block)")
                input_generate.append(f"        order = np.argsort(-v, kind='stable', axis=1)")
                input_generate.append(
                    f"        {name}__values = np.take_along_axis(v, order, axis=1).reshape(-1)[:{name}__struct_count]"
                )
                input_generate.append(
                    f"        {name}__idx = np.take_along_axis(i, order, axis=1).reshape(-1)[:{name}__struct_count]"
                )
            else:
                input_generate.append(f"    if {name}__struct_count > 0:")
                input_generate.append(f"        order = np.argsort(-{name}__values, kind='stable')")
                input_generate.append(f"        {name}__values = {name}__values[order]")
                input_generate.append(f"        {name}__idx = {name}__idx[order]")
            input_generate.append(f"    {name}__packed = np.empty(({name}__struct_count,), dtype={name}__struct_dtype)")
            input_generate.append(f"    {name}__packed['v'] = {name}__values")
            if np_dtype == "np.float16":
                input_generate.append(f"    {name}__packed['pad'] = np.uint16(0)")
            input_generate.append(f"    {name}__packed['i'] = {name}__idx")
            input_generate.append(f"    {name}__packed.tofile(\"{name}.bin\")")
        elif np_dtype.startswith("np.int") or np_dtype.startswith("np.uint"):
            if index_mod is not None:
                input_generate.append(
                    f"    {name} = (np.arange({size}, dtype=np.int64) % {index_mod}).astype({np_dtype})"
                )
            else:
                input_generate.append(f"    {name} = np.zeros(({size},), dtype={np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        else:
            input_generate.append(f"    {name} = np.random.random(size=({size},)).astype({np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")

    golden_py = golden_template.replace("@INPUT_GENERATE@", "\n".join(input_generate))
    (output_dir / "golden.py").write_text(golden_py, encoding="utf-8")

    # Emit the kernel source, optionally injecting a packed-predicate preload to
    # make TCMP/TCMPS outputs deterministic for byte-wise compares.
    kernel_text_out = raw_kernel_for_analysis
    if has_packed_pred_mask and output_ptrs:
        # Only handle the common packed-mask case (u8 output).
        mask_out = next((p for p in output_ptrs if p["cpp_type"] == "uint8_t"), None)
        if mask_out is not None:
            m = re.search(r"\bTCMPS?\s*\(\s*(\w+)\s*,", raw_kernel_for_analysis)
            if m:
                kernel_text_out = _inject_packed_pred_mask_preload(
                    kernel_text_out,
                    dst_tile=m.group(1),
                    output_ptr=mask_out["name"],
                    output_cpp_type=mask_out["cpp_type"],
                    rows=rows,
                    cols=cols,
                    logical_elem_count=logical_elem_count,
                )

    kernel_out = output_dir / f"{testcase}_kernel.cpp"
    kernel_out.write_text(_replace_includes(kernel_text_out), encoding="utf-8")

    launch_fn_params = ", ".join(launch_decl_params + ["void *stream"])
    kernel_call_args = []
    for p in params:
        if p["kind"] == "ptr":
            kernel_call_args.append(f"({_strip_param_name(p['raw'], p['name'])}){p['name']}")
        else:
            kernel_call_args.append(p["name"])
    kernel_call_args = ", ".join(kernel_call_args)
    launch_cpp = (
        INCLUDE_REPLACEMENT
        + "\n"
        f"__global__ AICORE void {kernel_name}({', '.join(raw_params)});\n\n"
        f"void {launch_name}({launch_fn_params}) {{\n"
        f"    {kernel_name}<<<1, nullptr, stream>>>({kernel_call_args});\n"
        f"}}\n"
    )
    (output_dir / "launch.cpp").write_text(launch_cpp, encoding="utf-8")

    mem_base_define = "MEMORY_BASE"
    if "910b" in (soc_version or "").lower():
        mem_base_define = "REGISTER_BASE"

    cce_stack_size_opt = ""
    # `-mllvm -cce-aicore-stack-size=...` is rejected on some targets (e.g.
    # dav-l310 / dav-l311).
    if not aicore_arch.startswith(("dav-l310", "dav-l311")):
        cce_stack_size_opt = '    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"\n'

    cmake_content = f"""
cmake_minimum_required(VERSION 3.16)

# Prefer setting compilers before project() so CMake picks up bisheng correctly.
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project({testcase}_npu_validation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910)
endif()
option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please source the CANN set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
endif()

set(PTO_ISA_ROOT "" CACHE PATH "Path to pto-isa repo")
if(NOT PTO_ISA_ROOT)
    set(_PTO_ISA_CANDIDATES
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../pto-isa"
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../pto-isa"
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../../pto-isa"
    )
    foreach(_cand IN LISTS _PTO_ISA_CANDIDATES)
        if(EXISTS "${{_cand}}/include" AND EXISTS "${{_cand}}/tests/common")
            set(PTO_ISA_ROOT "${{_cand}}" CACHE PATH "Path to pto-isa repo" FORCE)
            break()
        endif()
    endforeach()
endif()
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR "Cannot find PTO_ISA_ROOT, please pass -DPTO_ISA_ROOT=/path/to/pto-isa.")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
    -fPIC
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -fenable-matrix
    --cce-aicore-enable-tl
    -fPIC
    -Xhost-start -Xhost-end
{cce_stack_size_opt}\
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/tests/common
    ${{ASCEND_HOME_PATH}}/include
    ${{ASCEND_DRIVER_PATH}}/kernel/inc
)

	add_library({testcase}_kernel SHARED {testcase}_kernel.cpp launch.cpp)
	target_compile_options({testcase}_kernel PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} --cce-aicore-arch={aicore_arch}{dav_defines} -D{mem_base_define} -std=c++17)
	target_include_directories({testcase}_kernel PRIVATE
	    ${{ASCEND_HOME_PATH}}/pkg_inc/
	    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling/
	    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
	)
target_link_options({testcase}_kernel PRIVATE --cce-fatobj-link)

add_executable({testcase} main.cpp)
target_compile_options({testcase} PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
target_include_directories({testcase} PRIVATE
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/tests/common
)

target_link_directories({testcase} PUBLIC
    ${{ASCEND_HOME_PATH}}/lib64
)

target_link_libraries({testcase} PRIVATE
    {testcase}_kernel
    runtime
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)

if(ENABLE_SIM_GOLDEN)
    # Simulator executable: used to generate golden outputs (Ascend camodel).
    add_executable({testcase}_sim main.cpp)
    target_compile_options({testcase}_sim PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
    target_include_directories({testcase}_sim PRIVATE
        ${{PTO_ISA_ROOT}}/include
        ${{PTO_ISA_ROOT}}/tests/common
    )
    target_link_directories({testcase}_sim PUBLIC
        ${{ASCEND_HOME_PATH}}/lib64
        ${{ASCEND_HOME_PATH}}/aarch64-linux/simulator/${{SOC_VERSION}}/lib
        ${{ASCEND_HOME_PATH}}/simulator/${{SOC_VERSION}}/lib
        ${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib
    )
    target_link_libraries({testcase}_sim PRIVATE
        {testcase}_kernel
        runtime_camodel
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
    )
endif()
"""
    (output_dir / "CMakeLists.txt").write_text(cmake_content.strip() + "\n", encoding="utf-8")

    compare_template = (templates_root / "compare_template.py").read_text(encoding="utf-8")
    compare_lines = ["    ok = True"]
    compare_prefix_counts = {}
    for p in output_ptrs:
        name = p["name"]
        req = inferred_counts.get(name)
        if req is None:
            continue
        try:
            req = int(req)
        except Exception:
            continue
        if req <= 0:
            continue
        file_cnt = ptr_elem_counts.get(name, logical_elem_count)
        if file_cnt and req < int(file_cnt):
            compare_prefix_counts[name] = req
    for p in output_ptrs:
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        name = p["name"]
        eps = _default_eps_for_cpp_type(p["cpp_type"])
        if has_packed_pred_mask and p["cpp_type"] in {"uint8_t", "int8_t"}:
            compare_lines.append(
                f"    ok = compare_packed_pred_mask(\"golden_{name}.bin\", \"{name}.bin\", {rows}, {cols}) and ok"
            )
        else:
            prefix_cnt = compare_prefix_counts.get(name)
            if prefix_cnt is not None:
                compare_lines.append(
                    f"    ok = compare_bin_prefix(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, {eps}, {prefix_cnt}) and ok"
                )
            else:
                compare_lines.append(
                    f"    ok = compare_bin(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, {eps}) and ok"
                )
    compare_py = compare_template.replace("@COMPARES@", "\n".join(compare_lines))
    (output_dir / "compare.py").write_text(compare_py, encoding="utf-8")

    # Let the runner know which bins are outputs (for sim->golden copying).
    (output_dir / "outputs.txt").write_text(
        "\n".join([p["name"] for p in output_ptrs]) + ("\n" if output_ptrs else ""),
        encoding="utf-8",
    )

    run_sh = (templates_root / "run_sh_template.sh").read_text(encoding="utf-8")
    run_sh = run_sh.replace("@EXECUTABLE@", testcase)
    run_sh = run_sh.replace("@RUN_MODE@", run_mode)
    run_sh = run_sh.replace("@SOC_VERSION@", soc_version)
    run_path = output_dir / "run.sh"
    run_path.write_text(run_sh, encoding="utf-8")
    run_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description="Generate NPU validation testcase from PTOAS kernel.")
    parser.add_argument("--input", required=True, help="Input PTOAS .cpp file")
    parser.add_argument("--testcase", default=None, help="Testcase name (default: derived from input filename)")
    parser.add_argument("--output-root", default=None, help="Output testcases root directory")
    parser.add_argument("--run-mode", default="npu", choices=["sim", "npu"], help="Run mode for run.sh")
    parser.add_argument("--soc-version", default="Ascend910", help="SOC version for run.sh")
    parser.add_argument(
        "--aicore-arch",
        default=None,
        help="Override AICore arch passed to bisheng (e.g. dav-c220-vec|dav-c220-cube|dav-c310-vec|dav-c310-cube)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    testcase = args.testcase or _derive_testcase_name(Path(args.input))
    generate_testcase(
        Path(args.input),
        output_root,
        testcase,
        args.run_mode,
        args.soc_version,
        aicore_arch=args.aicore_arch,
    )


if __name__ == "__main__":
    main()
