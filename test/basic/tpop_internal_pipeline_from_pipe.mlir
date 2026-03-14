// RUN: ptoas %s | FileCheck %s

module {
  func.func @tpop_internal_pipeline_from_pipe(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %local_fifo_addr: i32) {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index

    %vec_mem = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile = pto.bind_tile %vec_mem, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1, local_fifo_depth = 4}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>, %local_fifo_addr : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      %fifo_tile = pto.declare_tile -> memref<32x128xf32, #pto.address_space<vec>>
      pto.tpop_internal(%fifo_tile, %pipe : memref<32x128xf32, #pto.address_space<vec>>, !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)
      pto.tmov ins(%fifo_tile : memref<32x128xf32, #pto.address_space<vec>>) outs(%vec_tile : memref<32x128xf32, #pto.address_space<vec>>)
      pto.tfree_internal(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)
    }
    return
  }
}

// CHECK: __global__ AICORE void tpop_internal_pipeline_from_pipe
// CHECK: TPOP(
// CHECK: TMOV(
// CHECK: TFREE(
