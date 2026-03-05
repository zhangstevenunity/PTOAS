// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @loop_outer_live(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                             %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // A buffer that remains live across the loop (used after the loop).
    %outer = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%outer : memref<16x16x16xf16, #pto.address_space<vec>>)

    // A loop-local buffer used inside the loop.
    %inner = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    scf.for %i = %c0 to %c4 step %c1 {
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%inner : memref<16x16x16xf16, #pto.address_space<vec>>)
      pto.tstore ins(%inner : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    }

    // Use %outer after the loop to keep it live across the loop.
    pto.tstore ins(%outer : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @loop_outer_live
// CHECK-NOT: memref.alloc
// Expect a loop, and two planned buffers at distinct offsets.
// CHECK-DAG: %c0_i64 = arith.constant 0 : i64
// CHECK-DAG: %c8192_i64 = arith.constant 8192 : i64
// CHECK-DAG: pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%c8192_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK: scf.for
