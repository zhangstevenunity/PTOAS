// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_bidirectional_dir_mask_not_supported(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %cube_tile: memref<64x128xf32, #pto.address_space<mat>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %fifo = pto.initialize_pipe {dir_mask = 3, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.fifo<memref<64x128xf32, #pto.address_space<mat>>, memref<64x128xf32, #pto.address_space<vec>>, #pto.fifo_location<gm>, 2, 2>

    pto.section.cube {
      pto.tpush(%cube_tile, %fifo : memref<64x128xf32, #pto.address_space<mat>>, !pto.fifo<memref<64x128xf32, #pto.address_space<mat>>, memref<64x128xf32, #pto.address_space<vec>>, #pto.fifo_location<gm>, 2, 2>)
    }
    return
  }
}

// CHECK: error: 'pto.initialize_pipe' op bidirectional DIRMASK is not supported
