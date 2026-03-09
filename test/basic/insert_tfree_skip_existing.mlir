// RUN: ptoas --pto-arch=a5 %s | FileCheck %s
//
// Verify that InsertTFreePass skips tpop ops that already have a hand-written tfree.

module {
  func.func @skip_existing_tfree(
      %gm_slot_buffer: memref<64x128xf32, #pto.address_space<gm>>,
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index

    %vec_mem = memref.alloc() : memref<32x128xf32, #pto.address_space<vec>>
    %vec_tile = pto.bind_tile %vec_mem, %c32, %c128 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x128xf32, #pto.address_space<vec>> -> memref<32x128xf32, #pto.address_space<vec>>

    %pipe = pto.initialize_pipe {dir_mask = 1, slot_size = 1024}
      (%gm_slot_buffer : memref<64x128xf32, #pto.address_space<gm>>,
       %c2v_consumer_buf : i32,
       %v2c_consumer_buf : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<mat>>, memref<32x128xf32, #pto.address_space<vec>>>

    pto.section.vector {
      pto.tpop(%vec_tile, %pipe : memref<32x128xf32, #pto.address_space<vec>>, !pto.pipe<memref<64x128xf32, #pto.address_space<mat>>, memref<32x128xf32, #pto.address_space<vec>>>)
      // Hand-written tfree — pass should skip this tpop
      pto.tfree(%pipe : !pto.pipe<memref<64x128xf32, #pto.address_space<mat>>, memref<32x128xf32, #pto.address_space<vec>>>)
    }

    return
  }
}

// Expect exactly one TFREE, not two
// CHECK: TPOP(
// CHECK: TFREE(
// CHECK-NOT: TFREE(
