PTO_IR = r"""

module {
  func.func @if_yield(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                      %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %cond = arith.constant true
    // Use scf.if control-flow without yielding a memref (the current emitc
    // lowering can't handle memref-typed scf.if results), but still ensure
    // PlanMemory rewrites allocs inside both branches.
    scf.if %cond {
      %then = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
               outs(%then : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tstore ins(%then : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    } else {
      %els = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
               outs(%els : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tstore ins(%els : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    }
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
