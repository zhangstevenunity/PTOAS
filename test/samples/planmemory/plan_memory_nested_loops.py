PTO_IR = r"""

module {
  func.func @nested_loops(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                          %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Outer buffer stays live across both loops (used after).
    %outer = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%outer : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    scf.for %i = %c0 to %c2 step %c1 {
      // Buffer allocated inside the outer loop.
      %mid = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
               outs(%mid : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      scf.for %j = %c0 to %c2 step %c1 {
        // A temp buffer in the inner loop.
        %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
        pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
                 outs(%tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.tstore ins(%tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                  outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
      }
      pto.tstore ins(%mid : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    }

    pto.tstore ins(%outer : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
