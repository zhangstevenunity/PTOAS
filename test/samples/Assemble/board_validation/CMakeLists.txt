cmake_minimum_required(VERSION 3.16)

# Prefer setting compilers before project() so CMake picks up bisheng correctly.
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project(assemble_npu_validation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910)
endif()
option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please source the CANN set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(PTO_ISA_ROOT "" CACHE PATH "Path to pto-isa repo")
if(NOT PTO_ISA_ROOT)
    set(_PTO_ISA_CANDIDATES
        "${CMAKE_CURRENT_LIST_DIR}/../../../../pto-isa"
        "${CMAKE_CURRENT_LIST_DIR}/../../../../../pto-isa"
        "${CMAKE_CURRENT_LIST_DIR}/../../../../../../pto-isa"
    )
    foreach(_cand IN LISTS _PTO_ISA_CANDIDATES)
        if(EXISTS "${_cand}/include" AND EXISTS "${_cand}/tests/common")
            set(PTO_ISA_ROOT "${_cand}" CACHE PATH "Path to pto-isa repo" FORCE)
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
    ${PTO_ISA_ROOT}/include
    ${PTO_ISA_ROOT}/tests/common
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

	add_library(assemble_kernel SHARED assemble_kernel.cpp launch.cpp)
	target_compile_options(assemble_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-cube -DMEMORY_BASE -std=c++17)
	target_include_directories(assemble_kernel PRIVATE
	    ${ASCEND_HOME_PATH}/pkg_inc/
	    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
	    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
	)
target_link_options(assemble_kernel PRIVATE --cce-fatobj-link)

add_executable(assemble main.cpp)
target_compile_options(assemble PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(assemble PRIVATE
    ${PTO_ISA_ROOT}/include
    ${PTO_ISA_ROOT}/tests/common
)

target_link_directories(assemble PUBLIC
    ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(assemble PRIVATE
    assemble_kernel
    runtime
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)

if(ENABLE_SIM_GOLDEN)
    # Simulator executable: used to generate golden outputs (Ascend camodel).
    add_executable(assemble_sim main.cpp)
    target_compile_options(assemble_sim PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(assemble_sim PRIVATE
        ${PTO_ISA_ROOT}/include
        ${PTO_ISA_ROOT}/tests/common
    )
    target_link_directories(assemble_sim PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )
    target_link_libraries(assemble_sim PRIVATE
        assemble_kernel
        runtime_camodel
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
    )
endif()
