// RUN: ptoas --pto-arch=a5 %s | FileCheck %s
//
// A pto.tpop followed by a compute op that reads the popped tile,
// but NO hand-written pto.tfree.  The InsertTFree pass should
// auto-insert pto.tfree after the last use of vec_tile.

module {
  func.func @auto_tfree(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index

    %vec_mem = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile = pto.bind_tile %vec_mem, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    // A second vec tile for the abs destination
    %vec_mem2 = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile2 = pto.bind_tile %vec_mem2, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_pipe {dir_mask = 1, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      // Pop data into vec_tile
      pto.tpop(%vec_tile, %pipe : memref<32x128xf32, #pto.address_space<vec>>, !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)

      // Compute op that reads vec_tile — this is the last use
      pto.tabs ins(%vec_tile : memref<32x128xf32, #pto.address_space<vec>>) outs(%vec_tile2 : memref<32x128xf32, #pto.address_space<vec>>)

      // NO pto.tfree here — the pass should insert it automatically
    }

    return
  }
}

// CHECK: TPOP(
// CHECK: TABS(
// CHECK: TFREE(
