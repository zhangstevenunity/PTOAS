module {
  func.func @matmul_tpush_tpop_print(
      %gm_a: memref<16x16xf32, #pto.address_space<gm>>,
      %gm_b: memref<16x16xf32, #pto.address_space<gm>>,
      %c2v_consumer_buf: i32) {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index

    %mat_a_mem = memref.alloc() : memref<16x16xf32, #pto.address_space<mat>>
    %mat_b_mem = memref.alloc() : memref<16x16xf32, #pto.address_space<mat>>
    %left_mem = memref.alloc() : memref<16x16xf32, #pto.address_space<left>>
    %right_mem = memref.alloc() : memref<16x16xf32, #pto.address_space<right>>
    %acc_mem = memref.alloc() : memref<16x16xf32, #pto.address_space<acc>>

    // TileBuf-only tprint path (currently may be folded out by some lowering passes).
    %vec_print = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %mat_a_tile = pto.bind_tile %mat_a_mem, %c16, %c16 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<16x16xf32, #pto.address_space<mat>> -> memref<16x16xf32, #pto.address_space<mat>>
    %mat_b_tile = pto.bind_tile %mat_b_mem, %c16, %c16 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<16x16xf32, #pto.address_space<mat>> -> memref<16x16xf32, #pto.address_space<mat>>

    %left_tile = pto.bind_tile %left_mem, %c16, %c16 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<16x16xf32, #pto.address_space<left>> -> memref<16x16xf32, #pto.address_space<left>>
    %right_tile = pto.bind_tile %right_mem, %c16, %c16 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<col_major>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<16x16xf32, #pto.address_space<right>> -> memref<16x16xf32, #pto.address_space<right>>

    %acc_tile = pto.bind_tile %acc_mem, %c16, %c16 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>>
    } : memref<16x16xf32, #pto.address_space<acc>> -> memref<16x16xf32, #pto.address_space<acc>>

    %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<memref<16x16xf32, #pto.address_space<acc>>, memref<8x16xf32, #pto.address_space<vec>>>

    pto.section.cube {
      pto.tload ins(%gm_a : memref<16x16xf32, #pto.address_space<gm>>) outs(%mat_a_tile : memref<16x16xf32, #pto.address_space<mat>>)
      pto.tload ins(%gm_b : memref<16x16xf32, #pto.address_space<gm>>) outs(%mat_b_tile : memref<16x16xf32, #pto.address_space<mat>>)

      pto.tmov ins(%mat_a_tile : memref<16x16xf32, #pto.address_space<mat>>) outs(%left_tile : memref<16x16xf32, #pto.address_space<left>>)
      pto.tmov ins(%mat_b_tile : memref<16x16xf32, #pto.address_space<mat>>) outs(%right_tile : memref<16x16xf32, #pto.address_space<right>>)

      pto.tmatmul ins(%left_tile, %right_tile : memref<16x16xf32, #pto.address_space<left>>, memref<16x16xf32, #pto.address_space<right>>) outs(%acc_tile : memref<16x16xf32, #pto.address_space<acc>>)

      pto.tpush(%acc_tile, %pipe : memref<16x16xf32, #pto.address_space<acc>>, !pto.pipe<memref<16x16xf32, #pto.address_space<acc>>, memref<8x16xf32, #pto.address_space<vec>>>)
    }

    pto.section.vector {
      %slot_id = pto.tpop(%pipe : !pto.pipe<memref<16x16xf32, #pto.address_space<acc>>, memref<8x16xf32, #pto.address_space<vec>>>) -> index
      %fifo_tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<memref<16x16xf32, #pto.address_space<acc>>, memref<8x16xf32, #pto.address_space<vec>>>, index)
        -> memref<8x16xf32, #pto.address_space<vec>>
      pto.tmov ins(%fifo_tile : memref<8x16xf32, #pto.address_space<vec>>) outs(%vec_print : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tprint ins(%vec_print : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tfree(%pipe, %slot_id : !pto.pipe<memref<16x16xf32, #pto.address_space<acc>>, memref<8x16xf32, #pto.address_space<vec>>>, index)
    }

    return
  }
}
