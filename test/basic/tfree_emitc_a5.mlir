// RUN: ptoas --pto-arch=a5 %s | FileCheck %s

module {
  func.func @tfree_a5_bidirectional(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index

    %acc_mem = memref.alloc() : memref<64x128xf32, #pto.address_space<acc>>
    %cube_mem = memref.alloc() : memref<64x128xf32, #pto.address_space<mat>>
    %vec_mem = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %acc_tile = pto.bind_tile %acc_mem, %c64, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=1024, pad=#pto.pad_value<null>>
    } : memref<64x128xf32, #pto.address_space<acc>> -> memref<64x128xf32, #pto.address_space<acc>>
    %cube_tile = pto.bind_tile %cube_mem, %c64, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<col_major>, slayout=#pto.slayout<row_major>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<64x128xf32, #pto.address_space<mat>> -> memref<64x128xf32, #pto.address_space<mat>>
    %vec_tile = pto.bind_tile %vec_mem, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe_c2v = pto.initialize_pipe {dir_mask = 1, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>

    %pipe_v2c = pto.initialize_pipe {dir_mask = 2, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<32x128xf32, #pto.address_space<vec>>, memref<64x128xf32, #pto.address_space<mat>>>

    // C2V: acc pushes, vector pops + frees
    pto.section.cube {
      pto.tpush(%acc_tile, %pipe_c2v : memref<64x128xf32, #pto.address_space<acc>>, !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)
    }
    pto.section.vector {
      pto.tpop(%vec_tile, %pipe_c2v : memref<32x128xf32, #pto.address_space<vec>>, !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)
      pto.tfree(%pipe_c2v : !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>, memref<32x128xf32, #pto.address_space<vec>>>)
    }

    // V2C: vector pushes, cube pops + frees
    pto.section.vector {
      pto.tpush(%vec_tile, %pipe_v2c : memref<32x128xf32, #pto.address_space<vec>>, !pto.pipe<memref<32x128xf32, #pto.address_space<vec>>, memref<64x128xf32, #pto.address_space<mat>>>)
    }
    pto.section.cube {
      pto.tpop(%cube_tile, %pipe_v2c : memref<64x128xf32, #pto.address_space<mat>>, !pto.pipe<memref<32x128xf32, #pto.address_space<vec>>, memref<64x128xf32, #pto.address_space<mat>>>)
      pto.tfree(%pipe_v2c : !pto.pipe<memref<32x128xf32, #pto.address_space<vec>>, memref<64x128xf32, #pto.address_space<mat>>>)
    }

    return
  }
}

// CHECK: TPipe<{{.*}}Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024
// CHECK: Tile<TileType::Mat, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null>
// CHECK: TPUSH(
// CHECK: TPOP(
// CHECK: TFREE(
// CHECK: TPUSH(
// CHECK: TPOP(
// CHECK: TFREE(
