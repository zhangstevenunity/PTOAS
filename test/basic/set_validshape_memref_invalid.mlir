// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @memref_arg_case(
      %arg0: memref<32x32xf32, #pto.address_space<vec>>) {
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    pto.set_validshape %arg0, %c16, %c24
      : memref<32x32xf32, #pto.address_space<vec>>
    return
  }
}

// CHECK: error: 'pto.set_validshape' op expects tile_buf source; memref source is only valid for the internal lowered form
