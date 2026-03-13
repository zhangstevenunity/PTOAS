// RUN: not ptoas --pto-arch=a5 %s 2>&1 | FileCheck %s
//
// Explicit tfree must appear after the last use of the borrowed tile.

module {
  func.func @tfree_before_last_use(
      %c2v_consumer_buf: i32) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index

    %vec_mem0 = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile0 = pto.bind_tile %vec_mem0, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      %slot_id = pto.tpop(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>) -> index
      %fifo_tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>

      pto.tfree(%pipe, %slot_id : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
      pto.tmov ins(%fifo_tile : memref<32x128xf32, #pto.address_space<vec>>) outs(%vec_tile0 : memref<32x128xf32, #pto.address_space<vec>>)
    }

    return
  }
}

// CHECK: error:
// CHECK: tfree_internal must appear after the last use of the borrowed tile
