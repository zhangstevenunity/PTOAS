// RUN: not ptoas --pto-arch=a5 %s 2>&1 | FileCheck %s
//
// One pipe may not have two outstanding borrowed tiles at the same time, even
// when both slots eventually have explicit tfree.

module {
  func.func @multiple_outstanding_same_pipe(
      %c2v_consumer_buf: i32) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index

    %vec_mem0 = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile0 = pto.bind_tile %vec_mem0, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %vec_mem1 = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile1 = pto.bind_tile %vec_mem1, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      %slot_id0 = pto.tpop(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>) -> index
      %fifo_tile0 = pto.get_fifo_tile(%pipe, %slot_id0 : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>

      %slot_id1 = pto.tpop(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>) -> index
      %fifo_tile1 = pto.get_fifo_tile(%pipe, %slot_id1 : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>

      // The first borrowed tile stays live past the second pop.
      pto.tabs ins(%fifo_tile0 : memref<32x128xf32, #pto.address_space<vec>>) outs(%vec_tile0 : memref<32x128xf32, #pto.address_space<vec>>)
      pto.tfree(%pipe, %slot_id0 : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
      pto.tmov ins(%fifo_tile1 : memref<32x128xf32, #pto.address_space<vec>>) outs(%vec_tile1 : memref<32x128xf32, #pto.address_space<vec>>)
      pto.tfree(%pipe, %slot_id1 : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
    }

    return
  }
}

// CHECK: error:
// CHECK: multiple outstanding pops on the same pipe are not supported
