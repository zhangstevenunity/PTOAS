def main():
    # Reproducer for zhangstevenunity/PTOAS#128:
    # `ptoas --enable-insert-sync` must handle double-buffer + dynamic-shape IR
    # without crashing in SyncEventIdAllocation.
    print(
        r"""module {
  func.func @vec_add_1d_dynamic(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f32>, %arg3: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8192 = arith.constant 8192 : index
    %0 = pto.get_block_idx
    %1 = pto.get_subblock_idx
    %2 = pto.get_subblock_num
    %3 = arith.muli %0, %2 : i64
    %4 = arith.addi %3, %1 : i64
    %5 = pto.get_block_num
    %6 = arith.index_cast %4 : i64 to index
    %7 = arith.index_cast %5 : i64 to index
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = arith.ceildivsi %8, %c8192 : index
    %10 = arith.ceildivsi %9, %7 : index
    %11 = arith.muli %6, %10 : index
    pto.section.vector {
      %12 = pto.make_tensor_view %arg0, shape = [%8] strides = [%c1] : !pto.tensor_view<?xf32>
      %13 = pto.make_tensor_view %arg1, shape = [%8] strides = [%c1] : !pto.tensor_view<?xf32>
      %14 = pto.make_tensor_view %arg2, shape = [%8] strides = [%c1] : !pto.tensor_view<?xf32>
      %15 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %16 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %17 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %18 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %19 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %20 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      %21 = arith.cmpi slt, %11, %9 : index
      scf.if %21 {
        %22 = arith.addi %11, %10 : index
        %23 = arith.cmpi sgt, %22, %9 : index
        %24 = arith.subi %9, %11 : index
        %25 = arith.select %23, %24, %10 : index
        %26 = arith.muli %25, %c8192 : index
        %27 = arith.cmpi sgt, %26, %c0 : index
        scf.if %27 {
          scf.for %arg4 = %c0 to %25 step %c1 {
            %28 = arith.addi %arg4, %11 : index
            %29 = arith.muli %28, %c8192 : index
            %30 = pto.partition_view %12, offsets = [%29], sizes = [%c8192] : !pto.tensor_view<?xf32> -> !pto.partition_tensor_view<1x8192xf32>
            %31 = pto.partition_view %13, offsets = [%29], sizes = [%c8192] : !pto.tensor_view<?xf32> -> !pto.partition_tensor_view<1x8192xf32>
            %32 = pto.partition_view %14, offsets = [%29], sizes = [%c8192] : !pto.tensor_view<?xf32> -> !pto.partition_tensor_view<1x8192xf32>
            %33 = arith.remui %arg4, %c2 : index
            %34 = arith.cmpi eq, %33, %c0 : index
            scf.if %34 {
              pto.tload ins(%30 : !pto.partition_tensor_view<1x8192xf32>) outs(%15 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              pto.tload ins(%31 : !pto.partition_tensor_view<1x8192xf32>) outs(%16 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              pto.tadd ins(%15, %16 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%17 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              pto.tstore ins(%17 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%32 : !pto.partition_tensor_view<1x8192xf32>)
            }
            %35 = arith.cmpi eq, %33, %c1 : index
            scf.if %35 {
              pto.tload ins(%30 : !pto.partition_tensor_view<1x8192xf32>) outs(%18 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              pto.tload ins(%31 : !pto.partition_tensor_view<1x8192xf32>) outs(%19 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              pto.tadd ins(%18, %19 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%20 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              pto.tstore ins(%20 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8192, v_row=1, v_col=8192, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%32 : !pto.partition_tensor_view<1x8192xf32>)
            }
          }
        }
      }
    }
    return
  }
}
"""
    )


if __name__ == "__main__":
    main()

