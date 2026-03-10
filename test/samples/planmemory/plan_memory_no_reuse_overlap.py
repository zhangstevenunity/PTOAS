PTO_IR = r"""

module {
  func.func @no_reuse_overlap(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                              %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %ub0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %ub1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    // Make lifetimes overlap by using both buffers after both are created.
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    pto.tstore ins(%ub1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    return
  }
}

// With overlapping lifetimes, offsets must differ.
"""

if __name__ == "__main__":
    print(PTO_IR)
