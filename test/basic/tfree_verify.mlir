// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @tfree_outside_section(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %pipe = pto.initialize_pipe {dir_mask = 1, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    // tfree outside section — should fail
    pto.tfree(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)
    return
  }
}

// CHECK: error: 'pto.tfree' op must be inside a section.cube or section.vector
