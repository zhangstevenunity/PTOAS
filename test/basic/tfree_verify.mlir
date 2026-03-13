// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @tfree_outside_section(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    // tfree outside section — should fail
    pto.tfree(%pipe, %c0 : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
    return
  }
}

// CHECK: error: 'pto.tfree' op must be inside a section.cube or section.vector
