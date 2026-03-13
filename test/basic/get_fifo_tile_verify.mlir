// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_get_fifo_tile_pipe_mismatch(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %local_addr0: i32,
      %local_addr1: i32) {
    %pipe0 = pto.initialize_l2g2l_pipe {dir_mask = 1}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>, %local_addr0 : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>
    %pipe1 = pto.initialize_l2l_pipe {dir_mask = 1}
      (%local_addr1 : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      %slot_id = pto.tpop(%pipe0 : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>) -> index
      %tile = pto.get_fifo_tile(%pipe1, %slot_id : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>
      pto.tfree(%pipe0, %slot_id : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
    }
    return
  }
}

// CHECK: error: 'pto.get_fifo_tile' op pipe_handle must match the pto.tpop that produced slot_id
