// RUN: ptoas %s | FileCheck %s

module {
  func.func @subset_infer_dynamic_offset(%off : index) {
    %c0 = arith.constant 0 : index
    %ws = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64, v_row=32, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sub = "pto.subset"(%ws, %c0, %off) {sizes = [32, 64]} :
      (!pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64, v_row=32, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index, index)
      -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64, v_row=32, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    return
  }
}

// CHECK: Success
