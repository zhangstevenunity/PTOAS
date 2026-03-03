// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @if_in_loop(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                        %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    scf.for %i = %c0 to %c2 step %c1 {
      %is0 = arith.cmpi eq, %i, %c0 : index
      scf.if %is0 {
        %a = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
        pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
                 outs(%a : memref<16x16x16xf16, #pto.address_space<vec>>)
        pto.tstore ins(%a : memref<16x16x16xf16, #pto.address_space<vec>>)
                  outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
      } else {
        %b = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
        pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
                 outs(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
        pto.tstore ins(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
                  outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
      }
    }
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @if_in_loop
// CHECK-NOT: memref.alloc
// CHECK-DAG: %c0_i64 = arith.constant 0 : i64
// CHECK-DAG: %c8192_i64 = arith.constant 8192 : i64
// CHECK: scf.for
// CHECK: scf.if
// CHECK: pto.pointer_cast
