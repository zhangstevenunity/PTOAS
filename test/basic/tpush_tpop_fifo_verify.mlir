// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_bidirectional_dir_mask_not_supported(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %acc_tile: memref<64x128xf32, #pto.address_space<acc>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %pipe = pto.initialize_pipe {dir_mask = 3, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<64x128xf32, #pto.address_space<vec>>>

    pto.section.cube {
      pto.tpush(%acc_tile, %pipe : memref<64x128xf32, #pto.address_space<acc>>, !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<64x128xf32, #pto.address_space<vec>>>)
    }
    return
  }
}

// CHECK: error: 'pto.initialize_pipe' op bidirectional DIRMASK is not supported
