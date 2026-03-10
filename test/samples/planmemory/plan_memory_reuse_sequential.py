PTO_IR = r"""

module {
  func.func @reuse_sequential(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                              %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    // Force reuse:
    // UB capacity (default) is 1572864 bits (196608 bytes). Each buffer here is
    // 16*16*16*f16 = 8192 bytes. Allocating 30 such buffers exceeds UB capacity
    // unless memory reuse is applied.
    %ub0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub2 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub2 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub3 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub3 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub3 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub4 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub4 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub4 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub5 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub5 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub5 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub6 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub6 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub6 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub7 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub7 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub7 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub8 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub8 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub8 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub9 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub9 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub9 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub10 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub10 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub10 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub11 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub11 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub11 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub12 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub12 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub12 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub13 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub13 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub13 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub14 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub14 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub14 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub15 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub15 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub15 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub16 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub16 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub16 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub17 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub17 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub17 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub18 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub18 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub18 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub19 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub19 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub19 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub20 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub20 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub20 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub21 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub21 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub21 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub22 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub22 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub22 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub23 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub23 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub23 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub24 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub24 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub24 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub25 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub25 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub25 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub26 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub26 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub26 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub27 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub27 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub27 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub28 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub28 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub28 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    %ub29 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%ub29 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%ub29 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    return
  }
}

// Anchor checks after the PlanMemory debug marker (ptoas prints the module
// before and after planning).
// Expect at least two distinct allocations to reuse offset 0.
"""

if __name__ == "__main__":
    print(PTO_IR)
