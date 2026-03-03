// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @if_yield(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                      %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %cond = arith.constant true
    // Use scf.if control-flow without yielding a memref (the current emitc
    // lowering can't handle memref-typed scf.if results), but still ensure
    // PlanMemory rewrites allocs inside both branches.
    scf.if %cond {
      %then = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%then : memref<16x16x16xf16, #pto.address_space<vec>>)
      pto.tstore ins(%then : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    } else {
      %els = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%els : memref<16x16x16xf16, #pto.address_space<vec>>)
      pto.tstore ins(%els : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    }
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @if_yield
// CHECK-NOT: memref.alloc
// CHECK: scf.if
// CHECK: pto.pointer_cast
// CHECK: } else {
// CHECK: pto.pointer_cast
