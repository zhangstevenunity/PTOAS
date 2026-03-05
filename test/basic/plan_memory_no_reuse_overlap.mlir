// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @no_reuse_overlap(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                              %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %ub0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %ub1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>

    // Make lifetimes overlap by using both buffers after both are created.
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%ub1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%ub1 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)

    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @no_reuse_overlap
// CHECK-NOT: memref.alloc
// With overlapping lifetimes, offsets must differ.
// CHECK-DAG: %c0_i64 = arith.constant 0 : i64
// CHECK-DAG: %c8192_i64 = arith.constant 8192 : i64
// CHECK-DAG: pto.pointer_cast(%c0_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
// CHECK-DAG: pto.pointer_cast(%c8192_i64) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>
