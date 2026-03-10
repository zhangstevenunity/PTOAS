PTO_IR = r"""

module {
  func.func @for_iter_args_yield(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                                 %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      // Two allocs inside the loop body exercise liveness within loops and
      // per-iteration memory planning for overlapping lifetimes.
      %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
               outs(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
               outs(%b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tstore ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
      pto.tstore ins(%b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    }
    return
  }
}

// After matching the loop header, require at least one planned buffer inside
// the loop body (the inner alloc becomes a pointer_cast).
"""

if __name__ == "__main__":
    print(PTO_IR)
