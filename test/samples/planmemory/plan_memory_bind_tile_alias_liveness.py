PTO_IR = r"""

module {
  func.func @bind_tile_alias_liveness(%arg0: memref<16x256xf16, #pto.address_space<gm>>,
                                      %arg1: memref<16x256xf16, #pto.address_space<gm>>) {
    %c16 = arith.constant 16 : index

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)

    // Using %a should keep %a live; %b must not reuse %a's offset.
    pto.tload ins(%arg0 : memref<16x256xf16, #pto.address_space<gm>>)
             outs(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<16x256xf16, #pto.address_space<gm>>)
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
