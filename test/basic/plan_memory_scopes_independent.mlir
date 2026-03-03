// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @scopes_independent(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %ub = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %l1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<mat>>

    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%l1 : memref<16x16x16xf16, #pto.address_space<mat>>)
    pto.tstore ins(%l1 : memref<16x16x16xf16, #pto.address_space<mat>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @scopes_independent
// CHECK-NOT: memref.alloc
// Offsets are planned per-scope, so both UB and L1 can start at 0.
// CHECK-DAG: %c0_i64 = arith.constant 0 : i64
// CHECK-DAG: pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<mat>>

