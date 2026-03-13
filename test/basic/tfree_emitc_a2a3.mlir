// RUN: ptoas %s | FileCheck %s

module {
  func.func @tfree_a2a3_l2g2l(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %local_fifo_addr: i32) {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index

    %vec_mem = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile = pto.bind_tile %vec_mem, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>, %local_fifo_addr : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      %slot_id = pto.tpop(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>) -> index
      %fifo_tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>
      pto.tmov ins(%fifo_tile : memref<32x128xf32, #pto.address_space<vec>>) outs(%vec_tile : memref<32x128xf32, #pto.address_space<vec>>)
      pto.tfree(%pipe, %slot_id : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>, index)
    }

    return
  }
}

// CHECK: TPipe<{{.*}}Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024
// CHECK: TPOP(
// CHECK: TFREE(
