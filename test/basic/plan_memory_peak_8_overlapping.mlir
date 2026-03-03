// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @peak_8_overlapping(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    // Peak liveness: 8 buffers live at once.
    %u0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u2 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u3 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u4 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u5 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u6 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u7 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>

    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u0 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u1 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u2 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u3 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u4 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u5 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u6 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u7 : memref<16x16x16xf16, #pto.address_space<vec>>)

    pto.tstore ins(%u0 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u1 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u2 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u3 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u4 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u5 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u6 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u7 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @peak_8_overlapping
// CHECK-NOT: memref.alloc
// 8 live buffers implies a max offset of 7*8192 = 57344 bytes.
// CHECK: %[[O57344:.*]] = arith.constant 57344 : i64
// CHECK: pto.pointer_cast(%[[O57344]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>

