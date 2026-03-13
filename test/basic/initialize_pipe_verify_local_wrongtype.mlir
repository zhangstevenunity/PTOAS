// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_local_wrong_fifo_type(
      %buf: memref<64x128xf32, #pto.address_space<vec>>) {
    %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%buf : memref<64x128xf32, #pto.address_space<vec>>)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>
    return
  }
}

// CHECK: error: 'pto.initialize_l2l_pipe' op local_addr must be i32 when provided
