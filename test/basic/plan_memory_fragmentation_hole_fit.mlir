// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @fragmentation_hole_fit(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                    %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    // Force a tight situation:
    //   - Keep 23 UB buffers live across the function (23 * 8192 = 188416 B).
    //   - Only 1 UB slot remains. Two short-lived buffers must reuse that slot.
    %k0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k2 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k3 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k4 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k5 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k6 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k7 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k8 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k9 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k10 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k11 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k12 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k13 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k14 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k15 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k16 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k17 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k18 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k19 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k20 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k21 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %k22 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>

    // Touch all k-buffers early so their lifetimes start before the temps.
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k2 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k3 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k4 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k5 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k6 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k7 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k8 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k9 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k10 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k11 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k12 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k13 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k14 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k15 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k16 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k17 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k18 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k19 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k20 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k21 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%k22 : memref<16x16x16xf16, #pto.address_space<vec>>)

    %t0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%t0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%t0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    %t1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%t1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%t1 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    // Keep all k-buffers live until the end.
    pto.tstore ins(%k0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k1 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k2 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k3 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k4 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k5 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k6 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k7 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k8 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k9 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k10 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k11 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k12 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k13 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k14 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k15 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k16 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k17 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k18 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k19 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k20 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k21 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%k22 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @fragmentation_hole_fit
// CHECK-NOT: memref.alloc
// With 23 live UB buffers, there is exactly one remaining 8192B slot. The two
// short-lived buffers must reuse that slot (offset 23*8192 = 188416).
// CHECK-DAG: %[[O188416:.*]] = arith.constant 188416 : i64
// CHECK-DAG: pto.pointer_cast(%[[O188416]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%[[O188416]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
