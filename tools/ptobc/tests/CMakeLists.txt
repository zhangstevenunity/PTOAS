# ptobc tests (ctest)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PTObc_TESTDATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../testdata")

add_test(NAME ptobc_stage9_e2e
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIRS=${PTObc_TESTDATA_DIR}:${CMAKE_SOURCE_DIR}/test/samples
    ${CMAKE_CURRENT_LIST_DIR}/stage9_e2e.sh
)

add_test(NAME ptobc_to_ptoas_smoke
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=$<TARGET_FILE:pto-opt>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ptobc_to_ptoas_smoke.sh
)

add_test(NAME ptobc_opcode_coverage_check
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_LIST_DIR}/opcode_coverage_check.py
    ${CMAKE_SOURCE_DIR}/include/PTO/IR/PTOOps.td
    ${CMAKE_SOURCE_DIR}/tools/ptobc/generated/ptobc_opcodes_v0.h
)
