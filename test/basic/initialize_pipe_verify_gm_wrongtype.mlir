// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_gm_wrong_fifo_type(%addr: i32) {
    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
      (%addr : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>
    return
  }
}

// CHECK: error: 'pto.initialize_l2g2l_pipe' op gm_addr must be memref with #pto.address_space<gm>
