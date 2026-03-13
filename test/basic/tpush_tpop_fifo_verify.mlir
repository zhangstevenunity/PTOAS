// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_bidirectional_dir_mask_not_supported(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %acc_tile: memref<64x128xf32, #pto.address_space<acc>>) {
    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 3}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<64x128xf32, #pto.address_space<vec>>>

    pto.section.cube {
      pto.tpush(%acc_tile, %pipe : memref<64x128xf32, #pto.address_space<acc>>, !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<64x128xf32, #pto.address_space<vec>>>)
    }
    return
  }
}

// CHECK: error: 'pto.initialize_l2g2l_pipe' op dir_mask must be 1 (C2V) or 2 (V2C)
