// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @fragmentation_two_holes(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                     %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    // Tight + overlap:
    //   - Keep 22 UB buffers live (22 * 8192 = 180224 B), leaving 2 free slots.
    //   - Allocate 2 short-lived buffers with overlapping lifetimes twice.
    // The two free slots are at offsets 22*8192 and 23*8192.
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

    // Stage 1: two overlapping temps.
    %a0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %b0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%a0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%b0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%a0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%b0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    // Stage 2: two overlapping temps again, which should reuse the same two slots.
    %a1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %b1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%a1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%b1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%a1 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%b1 : memref<16x16x16xf16, #pto.address_space<vec>>)
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
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @fragmentation_two_holes
// CHECK-NOT: memref.alloc
// With 22 live UB buffers, there are exactly two remaining 8192B slots at
// offsets 22*8192 and 23*8192, reused across the two stages.
// CHECK-DAG: %[[O180224:.*]] = arith.constant 180224 : i64
// CHECK-DAG: %[[O188416:.*]] = arith.constant 188416 : i64
// CHECK-DAG: pto.pointer_cast(%[[O180224]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%[[O180224]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%[[O188416]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%[[O188416]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
