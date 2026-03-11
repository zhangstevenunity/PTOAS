// RUN: ptoas %s | FileCheck %s

module {
  func.func @set_validshape_if_diff(%cond: i1) {
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %0 = pto.alloc_tile valid_row = %c32 valid_col = %c32
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    scf.if %cond {
      pto.set_validshape %0, %c16, %c24
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>
    } else {
      pto.set_validshape %0, %c8, %c12
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>
    }
    pto.tadd ins(%0, %0
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%0
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// CHECK: Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> [[ORIG:v[0-9]+]];
// CHECK: TASSIGN([[ORIG]],
// CHECK: Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> [[TILE:v[0-9]+]] = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(
// CHECK: __ubuf__ float* [[DATA:v[0-9]+]] = [[ORIG]].data();
// CHECK: TASSIGN([[TILE]],
// CHECK: if (
// CHECK: [[TILE]].RowMaskInternal = [[ROW1:v[0-9]+]];
// CHECK: [[TILE]].ColMaskInternal = [[COL1:v[0-9]+]];
// CHECK: } else {
// CHECK: [[TILE]].RowMaskInternal = [[ROW2:v[0-9]+]];
// CHECK: [[TILE]].ColMaskInternal = [[COL2:v[0-9]+]];
// CHECK: }
// CHECK: TADD([[TILE]], [[TILE]], [[TILE]]);
