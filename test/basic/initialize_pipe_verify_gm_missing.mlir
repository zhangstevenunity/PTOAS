// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_gm_missing_fifo_addr() {
    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>
    return
  }
}

// CHECK: error: expected '('
