// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @bind_tile_alias_liveness(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                      %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %c16 = arith.constant 16 : index

    %a = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %av = pto.bind_tile %a, %c16, %c16
      {config = #pto.tile_buf_config<blayout=1 : i32, slayout=1 : i32, s_fractal_size=512, pad=0 : i32>}
      : memref<16x16x16xf16, #pto.address_space<vec>> -> memref<16x16x16xf16, #pto.address_space<vec>>

    %b = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    // Using %av should keep %a live; %b must not reuse %a's offset.
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%av : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%av : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @bind_tile_alias_liveness
// CHECK-NOT: memref.alloc
// CHECK-DAG: %c0_i64 = arith.constant 0 : i64
// CHECK-DAG: %c8192_i64 = arith.constant 8192 : i64
// CHECK-DAG: pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%c8192_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>

