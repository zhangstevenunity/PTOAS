// RUN: ptoas %s | FileCheck %s

module {
  func.func @pipe_emitc(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index

    %cube_mem = memref.alloc() : memref<64x128xf32, #pto.address_space<mat>>
    %vec_mem = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %cube_tile = pto.bind_tile %cube_mem, %c64, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<64x128xf32, #pto.address_space<mat>> -> memref<64x128xf32, #pto.address_space<mat>>
    %vec_tile = pto.bind_tile %vec_mem, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_pipe {dir_mask = 1, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<mat>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.cube {
      pto.tpush(%cube_tile, %pipe : memref<64x128xf32, #pto.address_space<mat>>, !pto.pipe<memref<64x128xf32, #pto.address_space<mat>>, memref<32x128xf32, #pto.address_space<vec>>>)
    }

    pto.section.vector {
      pto.tpop(%vec_tile, %pipe : memref<32x128xf32, #pto.address_space<vec>>, !pto.pipe<memref<64x128xf32, #pto.address_space<mat>>, memref<32x128xf32, #pto.address_space<vec>>>)
    }
    return
  }
}

// CHECK: TPipe<
// CHECK-NOT: initialize_pipe<
// CHECK-NOT: initialize_pipe(
// CHECK-NOT: memref<
// CHECK-NOT: CrossCoreFIFO_
// CHECK-NOT: PTOASPipeHandle
// CHECK: Tile<TileType::Mat
// CHECK: Tile<TileType::Vec
// CHECK: TPUSH(
// CHECK: TPOP(
