// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @for_iter_args_yield(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                 %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      // Two allocs inside the loop body exercise liveness within loops and
      // per-iteration memory planning for overlapping lifetimes.
      %a = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
      %b = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%a : memref<16x16x16xf16, #pto.address_space<vec>>)
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
      pto.tstore ins(%a : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
      pto.tstore ins(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    }
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @for_iter_args_yield
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// After matching the loop header, require at least one planned buffer inside
// the loop body (the inner alloc becomes a pointer_cast).
// CHECK: pto.pointer_cast
// CHECK: pto.pointer_cast
