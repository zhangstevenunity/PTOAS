PTO_IR = r"""

module {
  func.func @loop_outer_live(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                             %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // A buffer that remains live across the loop (used after the loop).
    %outer = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%outer : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    // A loop-local buffer used inside the loop.
    %inner = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    scf.for %i = %c0 to %c4 step %c1 {
      pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
               outs(%inner : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tstore ins(%inner : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    }

    // Use %outer after the loop to keep it live across the loop.
    pto.tstore ins(%outer : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    return
  }
}

// Expect a loop, and two planned buffers at distinct offsets.
"""

if __name__ == "__main__":
    print(PTO_IR)
