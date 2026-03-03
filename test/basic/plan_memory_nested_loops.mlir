// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @nested_loops(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                          %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Outer buffer stays live across both loops (used after).
    %outer = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%outer : memref<16x16x16xf16, #pto.address_space<vec>>)

    scf.for %i = %c0 to %c2 step %c1 {
      // Buffer allocated inside the outer loop.
      %mid = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%mid : memref<16x16x16xf16, #pto.address_space<vec>>)
      scf.for %j = %c0 to %c2 step %c1 {
        // A temp buffer in the inner loop.
        %tmp = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
        pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
                 outs(%tmp : memref<16x16x16xf16, #pto.address_space<vec>>)
        pto.tstore ins(%tmp : memref<16x16x16xf16, #pto.address_space<vec>>)
                  outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
      }
      pto.tstore ins(%mid : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    }

    pto.tstore ins(%outer : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @nested_loops
// CHECK-NOT: memref.alloc
// CHECK-DAG: %c0_i64 = arith.constant 0 : i64
// CHECK-DAG: %c8192_i64 = arith.constant 8192 : i64
// CHECK: scf.for
// CHECK: scf.for
// CHECK: pto.pointer_cast
