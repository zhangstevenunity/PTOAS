#!/usr/bin/env python3
# coding=utf-8

import argparse
import re
from pathlib import Path

INCLUDE_REPLACEMENT = (
    "#include <pto/pto-inst.hpp>\n"
    "#include <pto/common/constants.hpp>\n"
    "#include \"acl/acl.h\"\n"
)


def _parse_shape(text: str):
    match = re.search(r"Shape<(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"Shape<\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 32, 32


def _extract_cpp_type(param: str) -> str:
    match = re.search(r"__gm__\s+([A-Za-z_]\w*)", param)
    return match.group(1) if match else "float"

def _extract_cpp_name(param: str) -> str:
    parts = param.strip().split()
    if not parts:
        return "arg"
    name = parts[-1].replace("*", "").strip()
    if name.startswith("__"):
        return "arg"
    return name


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
        "int8_t": "np.int8",
        "uint8_t": "np.uint8",
        "int16_t": "np.int16",
        "uint16_t": "np.uint16",
        "int32_t": "np.int32",
        "uint32_t": "np.uint32",
    }
    return mapping.get(cpp_type, "np.float32")


def _cpp_host_type(cpp_type: str) -> str:
    if cpp_type == "half":
        return "aclFloat16"
    return cpp_type


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


def generate_testcase(input_cpp: Path, output_root: Path, testcase: str, run_mode: str, soc_version: str):
    sample_dir = input_cpp.parent
    output_dir = sample_dir / "npu_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_out = output_dir / f"{testcase}_kernel.cpp"
    raw_kernel = input_cpp.read_text(encoding="utf-8")
    kernel_out.write_text(_replace_includes(raw_kernel), encoding="utf-8")

    rows, cols = _parse_shape(raw_kernel)
    raw_text = input_cpp.read_text(encoding="utf-8")
    kernel_name = _parse_kernel_name(raw_text)
    raw_params = _parse_kernel_params(raw_text)
    param_types = [_extract_cpp_type(p) for p in raw_params]
    param_names = [_extract_cpp_name(p) for p in raw_params]
    input_types = param_types[:-1] if param_types else ["float"]
    output_types = param_types[-1:] if param_types else ["float"]
    input_names_raw = param_names[:-1] if param_names else ["input1"]
    output_names_raw = param_names[-1:] if param_names else ["output1"]
    input_count = max(len(input_types), 1)
    output_count = max(len(output_types), 1)

    templates_root = Path(__file__).resolve().parents[1] / "templates"
    template = (templates_root / "main_template.cpp").read_text(encoding="utf-8")
    case_name = f"case_{rows}x{cols}"

    launch_decl_params = []
    launch_call_args = []
    alloc_host = []
    alloc_device = []
    read_inputs = []
    copy_inputs = []
    free_host = []
    free_device = []

    input_alloc_host = []
    input_alloc_device = []
    for idx in range(input_count):
        host = f"srcHost{idx}"
        dev = f"srcDevice{idx}"
        host_type = _cpp_host_type(input_types[idx] if idx < len(input_types) else "float")
        param_name = input_names_raw[idx] if idx < len(input_names_raw) else f"input{idx + 1}"
        launch_decl_params.append(f"{host_type} *{param_name}")
        launch_call_args.append(dev)
        alloc_host.append(f"    {host_type} *{host} = nullptr;")
        alloc_device.append(f"    {host_type} *{dev} = nullptr;")
        input_alloc_host.append(f"    aclrtMallocHost((void **)(&{host}), fileSize);")
        input_alloc_device.append(f"    aclrtMalloc((void **)&{dev}, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);")
        input_basename = input_names_raw[idx] if idx < len(input_names_raw) else f"input{idx + 1}"
        read_inputs.append(f"    ReadFile(\"./{input_basename}.bin\", fileSize, {host}, fileSize);")
        copy_inputs.append(
            f"    aclrtMemcpy({dev}, fileSize, {host}, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);"
        )
        free_host.append(f"    aclrtFreeHost({host});")
        free_device.append(f"    aclrtFree({dev});")

    output_decl = []
    output_alloc_host = []
    output_alloc_device = []
    output_copy_back = []
    output_write = []
    output_free_host = []
    output_free_device = []
    for idx in range(output_count):
        host = f"dstHost{idx}"
        dev = f"dstDevice{idx}"
        host_type = _cpp_host_type(output_types[idx] if idx < len(output_types) else "float")
        output_decl.append(f"    {host_type} *{host} = nullptr;")
        output_decl.append(f"    {host_type} *{dev} = nullptr;")
        output_alloc_host.append(f"    aclrtMallocHost((void **)(&{host}), fileSize);")
        output_alloc_device.append(f"    aclrtMalloc((void **)&{dev}, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);")
        output_copy_back.append(
            f"    aclrtMemcpy({host}, fileSize, {dev}, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);"
        )
        output_basename = output_names_raw[idx] if idx < len(output_names_raw) else f"output{idx + 1}"
        output_write.append(f"    WriteFile(\"./{output_basename}.bin\", {host}, fileSize);")
        output_free_device.append(f"    aclrtFree({dev});")
        output_free_host.append(f"    aclrtFreeHost({host});")

    output_param_name = output_names_raw[0] if output_names_raw else "output1"
    launch_decl_params.append(f"{_cpp_host_type(output_types[0])} *{output_param_name}")
    launch_call_args.append("dstDevice0")

    param_decls = "\n".join([
        f"    size_t fileSize = {rows} * {cols} * sizeof({_cpp_host_type(output_types[0])});",
    ] + output_decl + alloc_host + alloc_device)

    launch_name = f"Launch{kernel_name[0].upper()}{kernel_name[1:]}"
    main_cpp = (
        template
        .replace("@TEST_SUITE@", testcase.upper())
        .replace("@CASE_NAME@", case_name)
        .replace("@LAUNCH_DECL@", f"void {launch_name}({', '.join(launch_decl_params)}, void *stream);")
        .replace("@PARAM_DECLS@", param_decls)
        .replace("@ALLOC_HOST@", "\n".join(output_alloc_host + input_alloc_host))
        .replace("@ALLOC_DEVICE@", "\n".join(output_alloc_device + input_alloc_device))
        .replace("@READ_INPUTS@", "\n".join(read_inputs))
        .replace("@COPY_TO_DEVICE@", "\n".join(copy_inputs))
        .replace("@LAUNCH_CALL@", f"    {launch_name}({', '.join(launch_call_args)}, stream);")
        .replace("@COPY_BACK@", "\n".join(output_copy_back))
        .replace("@WRITE_OUTPUT@", "\n".join(output_write))
        .replace("@FREE_DEVICE@", "\n".join(free_device + output_free_device))
        .replace("@FREE_HOST@", "\n".join(free_host + output_free_host))
    )

    (output_dir / "main.cpp").write_text(main_cpp, encoding="utf-8")

    golden_template = (templates_root / "golden_template.py").read_text(encoding="utf-8")
    input_generate = []
    input_names = []
    for idx in range(input_count):
        np_dtype = _np_dtype_for_cpp(input_types[idx] if idx < len(input_types) else "float")
        name = input_names_raw[idx] if idx < len(input_names_raw) else f"input{idx + 1}"
        input_generate.append(f"    {name} = np.random.random(size=({rows * cols},)).astype({np_dtype})")
        input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        input_names.append(name)

    golden_outputs = []
    output_writes = []
    for idx in range(output_count):
        np_dtype = _np_dtype_for_cpp(output_types[idx] if idx < len(output_types) else "float")
        golden_outputs.append(
            f"    outputs.append(np.full({rows * cols}, -1, dtype={np_dtype}))"
        )
        name = output_names_raw[idx] if idx < len(output_names_raw) else f"output{idx + 1}"
        output_writes.append(f"    outputs[{idx}].tofile(\"golden_{name}.bin\")")

    golden_py = (
        golden_template
        .replace("@GOLDEN_ARGS@", ", ".join(input_names))
        .replace("@GOLDEN_CALL_ARGS@", ", ".join(input_names))
        .replace("@GOLDEN_RET@", "outputs")
        .replace("@INPUT_GENERATE@", "\n".join(input_generate))
        .replace("@GOLDEN_OUTPUTS@", "\n".join(golden_outputs))
        .replace("@OUTPUT_WRITES@", "\n".join(output_writes))
    )
    (output_dir / "golden.py").write_text(golden_py, encoding="utf-8")

    launch_args = [name.split()[-1].replace("*", "") for name in launch_decl_params]
    launch_cpp = (
        "#include <pto/pto-inst.hpp>\n"
        "#include <pto/common/constants.hpp>\n"
        "#include \"acl/acl.h\"\n\n"
        f"__global__ AICORE void {kernel_name}({', '.join(raw_params)});\n\n"
        f"void {launch_name}({', '.join(launch_decl_params)}, void *stream) {{\n"
        f"    {kernel_name}<<<1, nullptr, stream>>>({', '.join(launch_args)});\n"
        f"}}\n"
    )
    (output_dir / "launch.cpp").write_text(launch_cpp, encoding="utf-8")

    cmake_content = f"""
cmake_minimum_required(VERSION 3.16)
project({testcase}_npu_validation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE npu)
endif()
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910B1)
endif()

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run setenv.bash.")
else()
    set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
endif()

set(PTO_ISA_ROOT "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../../pto-isa" CACHE PATH "Path to pto-isa repo")
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${{CMAKE_COMPILER}})
set(CMAKE_CXX_COMPILER ${{CMAKE_COMPILER}})

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
    -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
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
target_compile_options({testcase}_kernel PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17)
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
    ${{ASCEND_HOME_PATH}}/simulator/${{SOC_VERSION}}/lib
    ${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib
)

target_link_libraries({testcase} PRIVATE
    {testcase}_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)
"""
    (output_dir / "CMakeLists.txt").write_text(cmake_content.strip() + "\n", encoding="utf-8")

    compare_template = (templates_root / "compare_template.py").read_text(encoding="utf-8")
    compare_lines = ["    ok = True"]
    for idx in range(output_count):
        np_dtype = _np_dtype_for_cpp(output_types[idx] if idx < len(output_types) else "float")
        name = output_names_raw[idx] if idx < len(output_names_raw) else f"output{idx + 1}"
        compare_lines.append(
            f"    ok = compare_bin(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, 0.0) and ok"
        )
    compare_py = compare_template.replace("@COMPARES@", "\n".join(compare_lines))
    (output_dir / "compare.py").write_text(compare_py, encoding="utf-8")

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
    parser.add_argument("--soc-version", default="Ascend910B1", help="SOC version for run.sh")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parents[1] / "testcases"
    testcase = args.testcase or _derive_testcase_name(Path(args.input))
    generate_testcase(Path(args.input), output_root, testcase, args.run_mode, args.soc_version)


if __name__ == "__main__":
    main()
