// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @peak_exact_capacity(%arg0: memref<16x16x16xf16, #pto.address_space<gm>>,
                                 %arg1: memref<16x16x16xf16, #pto.address_space<gm>>) {
    // Default UB size is 1572864 bits (196608 bytes). Each buffer here is
    // 16*16*16*f16 = 8192 bytes. 24 buffers live at once should fit exactly:
    //   24 * 8192 = 196608 bytes.
    %u0 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u1 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u2 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u3 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u4 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u5 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u6 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u7 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u8 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u9 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u10 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u11 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u12 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u13 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u14 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u15 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u16 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u17 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u18 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u19 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u20 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u21 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u22 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>
    %u23 = memref.alloc() : memref<16x16x16xf16, #pto.address_space<vec>>

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
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u8 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u9 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u10 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u11 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u12 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u13 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u14 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u15 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u16 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u17 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u18 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u19 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u20 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u21 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u22 : memref<16x16x16xf16, #pto.address_space<vec>>)
    pto.tload ins(%arg0 : memref<16x16x16xf16, #pto.address_space<gm>>)
             outs(%u23 : memref<16x16x16xf16, #pto.address_space<vec>>)

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
    pto.tstore ins(%u8 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u9 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u10 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u11 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u12 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u13 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u14 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u15 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u16 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u17 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u18 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u19 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u20 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u21 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u22 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    pto.tstore ins(%u23 : memref<16x16x16xf16, #pto.address_space<vec>>)
              outs(%arg1 : memref<16x16x16xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: end PTO plan Mem!
// CHECK: func.func @peak_exact_capacity
// CHECK-NOT: memref.alloc
// 24 live buffers implies a max offset of 23*8192 = 188416 bytes.
// CHECK: %[[O188416:.*]] = arith.constant 188416 : i64
// CHECK: pto.pointer_cast(%[[O188416]]) : memref<16x16x16xf16, #pto.address_space<{{vec|ub}}>>

