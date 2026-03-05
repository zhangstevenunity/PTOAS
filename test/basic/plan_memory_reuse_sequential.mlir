// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @reuse_sequential(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                              %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    // Force reuse:
    // UB capacity (default) is 1572864 bits (196608 bytes). Each buffer here is
    // 16*16*16*f16 = 8192 bytes. Allocating 30 such buffers exceeds UB capacity
    // unless memory reuse is applied.
    %ub0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub1 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub2 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub2 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub2 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub3 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub3 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub3 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub4 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub4 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub4 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub5 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub5 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub5 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub6 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub6 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub6 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub7 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub7 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub7 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub8 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub8 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub8 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub9 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub9 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub9 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub10 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub10 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub10 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub11 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub11 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub11 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub12 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub12 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub12 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub13 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub13 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub13 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub14 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub14 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub14 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub15 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub15 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub15 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub16 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub16 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub16 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub17 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub17 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub17 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub18 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub18 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub18 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub19 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub19 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub19 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub20 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub20 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub20 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub21 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub21 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub21 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub22 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub22 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub22 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub23 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub23 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub23 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub24 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub24 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub24 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub25 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub25 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub25 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub26 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub26 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub26 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub27 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub27 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub27 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub28 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub28 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub28 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %ub29 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub29 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub29 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    return
  }
}

// Anchor checks after the PlanMemory debug marker (ptoas prints the module
// before and after planning).
// CHECK: end PTO plan Mem!
// CHECK: func.func @reuse_sequential
// CHECK-NOT: memref.alloc
// Expect at least two distinct allocations to reuse offset 0.
// CHECK: %c0_i64 = arith.constant 0 : i64
// CHECK: %[[BUF0:.*]] = pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK: %[[BUF1:.*]] = pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
