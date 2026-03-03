// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @loop_in_if(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                        %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    scf.if %true {
      scf.for %i = %c0 to %c2 step %c1 {
        %a = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
        pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
                 outs(%a : memref<16x16x16xf16, #pto.address_space<vec>>)
        pto.tstore ins(%a : memref<16x16x16xf16, #pto.address_space<vec>>)
                  outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
      }
    } else {
      %b = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
      pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
               outs(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
      pto.tstore ins(%b : memref<16x16x16xf16, #pto.address_space<vec>>)
                outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    }
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @loop_in_if
// CHECK-NOT: memref.alloc
// CHECK: scf.if
// CHECK: scf.for
// CHECK: } else {
// CHECK: pto.pointer_cast

