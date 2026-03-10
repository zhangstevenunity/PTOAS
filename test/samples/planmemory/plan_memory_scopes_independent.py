PTO_IR = r"""

module {
  func.func @scopes_independent(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                                %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %ub = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %l1 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%l1 : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%l1 : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    return
  }
}

// Offsets are planned per-scope, so both UB and L1 can start at 0.
"""

if __name__ == "__main__":
    print(PTO_IR)
