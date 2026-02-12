cmake_minimum_required(VERSION 3.16)
project(matmul_vector_mix-pto-sync_npu_validation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE npu)
endif()
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910B1)
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run setenv.bash.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(PTO_ISA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../../pto-isa" CACHE PATH "Path to pto-isa repo")
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

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
    ${PTO_ISA_ROOT}/include
    ${PTO_ISA_ROOT}/tests/common
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

add_library(matmul_vector_mix-pto-sync_kernel SHARED matmul_vector_mix-pto-sync_kernel.cpp launch.cpp)
target_compile_options(matmul_vector_mix-pto-sync_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-cube -DMEMORY_BASE -std=c++17)
target_include_directories(matmul_vector_mix-pto-sync_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(matmul_vector_mix-pto-sync_kernel PRIVATE --cce-fatobj-link)

add_executable(matmul_vector_mix-pto-sync main.cpp)
target_compile_options(matmul_vector_mix-pto-sync PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(matmul_vector_mix-pto-sync PRIVATE
    ${PTO_ISA_ROOT}/include
    ${PTO_ISA_ROOT}/tests/common
)

target_link_directories(matmul_vector_mix-pto-sync PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(matmul_vector_mix-pto-sync PRIVATE
    matmul_vector_mix-pto-sync_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)
