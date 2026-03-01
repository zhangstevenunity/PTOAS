# Auto-generated test from upstream MLIR sample.

def build():
    return r"""
module {
  func.func @main_kernel(%arg0: !pto.ptr<f16>, %arg1: !pto.ptr<f16>, %arg2: !pto.ptr<i32>, %arg3: !pto.ptr<f16>, %arg4: !pto.ptr<f16>, %arg5: !pto.ptr<f32>, %arg6: !pto.ptr<f16>, %arg7: !pto.ptr<f32>, %arg8: i32, %arg9: i32, %arg10: i32) {
    %0 = pto.get_block_idx
    %1 = arith.trunci %0 : i64 to i32
    %2 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>
    %3 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>
    %4 = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %5 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=1>
    %6 = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %7 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %8 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %9 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %10 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %11 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %12 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %13 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %14 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %15 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=6144, v_row=1, v_col=6144, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %16 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %17 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %18 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %19 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %20 = pto.get_subblock_idx
    %21 = arith.trunci %20 : i64 to i32
    pto.section.cube {
      %22 = arith.divsi %1, %arg9 : i32
      %23 = arith.remsi %22, %arg8 : i32
      %24 = arith.muli %23, %arg9 : i32
      %c8192_i32 = arith.constant 8192 : i32
      %25 = arith.muli %24, %c8192_i32 : i32
      %26 = arith.remsi %1, %arg9 : i32
      %c8192_i32_0 = arith.constant 8192 : i32
      %27 = arith.muli %26, %c8192_i32_0 : i32
      %28 = arith.addi %25, %27 : i32
      %29 = arith.divsi %1, %arg9 : i32
      %30 = arith.divsi %29, %arg8 : i32
      %c4_i32 = arith.constant 4 : i32
      %31 = arith.remsi %30, %c4_i32 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %32 = arith.muli %31, %c2048_i32 : i32
      %33 = arith.addi %28, %32 : i32
      %c0_i32 = arith.constant 0 : i32
      %34 = arith.muli %arg8, %arg9 : i32
      %c8192_i32_1 = arith.constant 8192 : i32
      %35 = arith.muli %34, %c8192_i32_1 : i32
      %36 = arith.divsi %1, %arg9 : i32
      %37 = arith.divsi %36, %arg8 : i32
      %c4_i32_2 = arith.constant 4 : i32
      %38 = arith.remsi %37, %c4_i32_2 : i32
      %c2048_i32_3 = arith.constant 2048 : i32
      %39 = arith.muli %38, %c2048_i32_3 : i32
      %40 = arith.subi %35, %39 : i32
      %41 = arith.divsi %1, %arg9 : i32
      %42 = arith.remsi %41, %arg8 : i32
      %43 = arith.muli %42, %arg9 : i32
      %c8192_i32_4 = arith.constant 8192 : i32
      %44 = arith.muli %43, %c8192_i32_4 : i32
      %45 = arith.subi %40, %44 : i32
      %46 = arith.remsi %1, %arg9 : i32
      %c8192_i32_5 = arith.constant 8192 : i32
      %47 = arith.muli %46, %c8192_i32_5 : i32
      %48 = arith.subi %45, %47 : i32
      %c2048_i32_6 = arith.constant 2048 : i32
      %c1 = arith.constant 1 : index
      %49 = arith.index_cast %arg8 : i32 to index
      %50 = arith.index_cast %arg9 : i32 to index
      %c64_i32 = arith.constant 64 : i32
      %51 = arith.index_cast %c64_i32 : i32 to index
      %c128_i32 = arith.constant 128 : i32
      %52 = arith.index_cast %c128_i32 : i32 to index
      %c1_7 = arith.constant 1 : index
      %53 = arith.muli %c1_7, %52 : index
      %54 = arith.muli %53, %51 : index
      %55 = arith.muli %54, %50 : index
      %56 = arith.muli %55, %49 : index
      %57 = pto.make_tensor_view %arg0, shape = [%c1, %49, %50, %51, %52] strides = [%56, %55, %54, %53, %c1_7] : !pto.tensor_view<?x?x?x?x?xf16>
      %c0 = arith.constant 0 : index
      %58 = arith.index_cast %33 : i32 to index
      %c1_8 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c128 = arith.constant 128 : index
      %59 = pto.partition_view %57, offsets = [%c0, %c0, %c0, %c0, %58], sizes = [%c1_8, %c1_8, %c1_8, %c16, %c128] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<16x128xf16>
      %60 = pto.tload ins(%59 : !pto.partition_tensor_view<16x128xf16>) outs(%2 : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>) -> tensor<16x128xf16>
      pto.barrier <PIPE_ALL>
      %c0_i32_9 = arith.constant 0 : i32
      %61 = arith.index_cast %c0_i32_9 : i32 to index
      %c32_i32 = arith.constant 32 : i32
      %62 = arith.index_cast %c32_i32 : i32 to index
      %c1_10 = arith.constant 1 : index
      %63 = arith.addi %61, %62 : index
      scf.for %arg11 = %61 to %63 step %c1_10 {
        %64 = arith.index_cast %arg11 : index to i32
        pto.sync.wait <PIPE_MTE1>, 0
        pto.sync.wait <PIPE_MTE1>, 16
        pto.barrier <PIPE_ALL>
        %c8192_i32_11 = arith.constant 8192 : i32
        %65 = arith.muli %1, %c8192_i32_11 : i32
        %c0_i32_12 = arith.constant 0 : i32
        %c17891328_i32 = arith.constant 17891328 : i32
        %c8192_i32_13 = arith.constant 8192 : i32
        %66 = arith.muli %1, %c8192_i32_13 : i32
        %67 = arith.subi %c17891328_i32, %66 : i32
        %c8192_i32_14 = arith.constant 8192 : i32
        %c1_15 = arith.constant 1 : index
        %c2184_i32 = arith.constant 2184 : i32
        %68 = arith.index_cast %c2184_i32 : i32 to index
        %c64_i32_16 = arith.constant 64 : i32
        %69 = arith.index_cast %c64_i32_16 : i32 to index
        %c128_i32_17 = arith.constant 128 : i32
        %70 = arith.index_cast %c128_i32_17 : i32 to index
        %c1_18 = arith.constant 1 : index
        %71 = arith.muli %c1_18, %70 : index
        %72 = arith.muli %71, %69 : index
        %73 = arith.muli %72, %68 : index
        %74 = arith.muli %73, %c1_15 : index
        %75 = pto.make_tensor_view %arg4, shape = [%c1_15, %c1_15, %68, %69, %70] strides = [%74, %73, %72, %71, %c1_18] : !pto.tensor_view<?x?x?x?x?xf16>
        %c0_19 = arith.constant 0 : index
        %76 = arith.index_cast %65 : i32 to index
        %c1_20 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c128_21 = arith.constant 128 : index
        %77 = pto.partition_view %75, offsets = [%c0_19, %c0_19, %c0_19, %c0_19, %76], sizes = [%c1_20, %c1_20, %c1_20, %c64, %c128_21] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<64x128xf16>
        %78 = pto.tload ins(%77 : !pto.partition_tensor_view<64x128xf16>) outs(%3 : !pto.tile_buf<loc=mat, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>) -> tensor<64x128xf16>
        pto.barrier <PIPE_ALL>
        pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        %79 = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
        %80 = pto.tmov ins(%2 : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>) outs(%79 : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>) -> tensor<16x128xf16>
        %81 = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f16, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>
        pto.treshape ins(%3 : !pto.tile_buf<loc=mat, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>) outs(%81 : !pto.tile_buf<loc=mat, dtype=f16, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
        %82 = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>
        %83 = pto.tmov ins(%81 : !pto.tile_buf<loc=mat, dtype=f16, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%82 : !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>) -> tensor<128x64xf16>
        pto.set_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        pto.wait_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        %84 = pto.tmatmul ins(%79, %82 : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%4 : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) -> tensor<16x64xf32>
        pto.barrier <PIPE_ALL>
        %c0_i32_22 = arith.constant 0 : i32
        %c1024_i32 = arith.constant 1024 : i32
        %85 = arith.muli %1, %c1024_i32 : i32
        %c1024_i32_23 = arith.constant 1024 : i32
        %c2236416_i32 = arith.constant 2236416 : i32
        %c1024_i32_24 = arith.constant 1024 : i32
        %86 = arith.muli %1, %c1024_i32_24 : i32
        %87 = arith.subi %c2236416_i32, %86 : i32
        %c1_25 = arith.constant 1 : index
        %c2184_i32_26 = arith.constant 2184 : i32
        %88 = arith.index_cast %c2184_i32_26 : i32 to index
        %c16_i32 = arith.constant 16 : i32
        %89 = arith.index_cast %c16_i32 : i32 to index
        %c64_i32_27 = arith.constant 64 : i32
        %90 = arith.index_cast %c64_i32_27 : i32 to index
        %c1_28 = arith.constant 1 : index
        %91 = arith.muli %c1_28, %90 : index
        %92 = arith.muli %91, %89 : index
        %93 = arith.muli %92, %88 : index
        %94 = arith.muli %93, %c1_25 : index
        %95 = pto.make_tensor_view %arg5, shape = [%c1_25, %c1_25, %88, %89, %90] strides = [%94, %93, %92, %91, %c1_28] : !pto.tensor_view<?x?x?x?x?xf32>
        %c0_29 = arith.constant 0 : index
        %96 = arith.index_cast %85 : i32 to index
        %c1_30 = arith.constant 1 : index
        %c16_31 = arith.constant 16 : index
        %c64_32 = arith.constant 64 : index
        %97 = pto.partition_view %95, offsets = [%c0_29, %c0_29, %c0_29, %c0_29, %96], sizes = [%c1_30, %c1_30, %c1_30, %c16_31, %c64_32] : !pto.tensor_view<?x?x?x?x?xf32> -> !pto.partition_tensor_view<16x64xf32>
        %98 = pto.tstore ins(%4 : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) outs(%97 : !pto.partition_tensor_view<16x64xf32>) -> tensor<16x64xf32>
        pto.barrier <PIPE_ALL>
        pto.sync.set <PIPE_FIX>, 1
        pto.sync.set <PIPE_FIX>, 17
        pto.sync.wait <PIPE_MTE1>, 2
        pto.sync.wait <PIPE_MTE1>, 18
        pto.barrier <PIPE_ALL>
        %c1024_i32_33 = arith.constant 1024 : i32
        %99 = arith.muli %1, %c1024_i32_33 : i32
        %c0_i32_34 = arith.constant 0 : i32
        %c2236416_i32_35 = arith.constant 2236416 : i32
        %c1024_i32_36 = arith.constant 1024 : i32
        %100 = arith.muli %1, %c1024_i32_36 : i32
        %101 = arith.subi %c2236416_i32_35, %100 : i32
        %c1024_i32_37 = arith.constant 1024 : i32
        %c1_38 = arith.constant 1 : index
        %c2184_i32_39 = arith.constant 2184 : i32
        %102 = arith.index_cast %c2184_i32_39 : i32 to index
        %c16_i32_40 = arith.constant 16 : i32
        %103 = arith.index_cast %c16_i32_40 : i32 to index
        %c64_i32_41 = arith.constant 64 : i32
        %104 = arith.index_cast %c64_i32_41 : i32 to index
        %c1_42 = arith.constant 1 : index
        %105 = arith.muli %c1_42, %104 : index
        %106 = arith.muli %105, %103 : index
        %107 = arith.muli %106, %102 : index
        %108 = arith.muli %107, %c1_38 : index
        %109 = pto.make_tensor_view %arg6, shape = [%c1_38, %c1_38, %102, %103, %104] strides = [%108, %107, %106, %105, %c1_42] : !pto.tensor_view<?x?x?x?x?xf16>
        %c0_43 = arith.constant 0 : index
        %110 = arith.index_cast %99 : i32 to index
        %c1_44 = arith.constant 1 : index
        %c16_45 = arith.constant 16 : index
        %c64_46 = arith.constant 64 : index
        %111 = pto.partition_view %109, offsets = [%c0_43, %c0_43, %c0_43, %c0_43, %110], sizes = [%c1_44, %c1_44, %c1_44, %c16_45, %c64_46] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<16x64xf16>
        %112 = pto.tload ins(%111 : !pto.partition_tensor_view<16x64xf16>) outs(%5 : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=1>) -> tensor<16x64xf16>
        pto.barrier <PIPE_ALL>
        pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        %113 = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>
        %114 = pto.tmov ins(%5 : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=1>) outs(%113 : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>) -> tensor<16x64xf16>
        %115 = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>
        %116 = pto.tmov ins(%3 : !pto.tile_buf<loc=mat, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=1>) outs(%115 : !pto.tile_buf<loc=right, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>) -> tensor<64x128xf16>
        pto.set_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        pto.wait_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
        %117 = pto.tmatmul ins(%113, %115 : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=64, v_row=16, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%6 : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) -> tensor<16x128xf32>
        pto.barrier <PIPE_ALL>
        %c0_i32_47 = arith.constant 0 : i32
        %c2048_i32_48 = arith.constant 2048 : i32
        %118 = arith.muli %1, %c2048_i32_48 : i32
        %c2048_i32_49 = arith.constant 2048 : i32
        %c4472832_i32 = arith.constant 4472832 : i32
        %c2048_i32_50 = arith.constant 2048 : i32
        %119 = arith.muli %1, %c2048_i32_50 : i32
        %120 = arith.subi %c4472832_i32, %119 : i32
        %c1_51 = arith.constant 1 : index
        %c2184_i32_52 = arith.constant 2184 : i32
        %121 = arith.index_cast %c2184_i32_52 : i32 to index
        %c16_i32_53 = arith.constant 16 : i32
        %122 = arith.index_cast %c16_i32_53 : i32 to index
        %c128_i32_54 = arith.constant 128 : i32
        %123 = arith.index_cast %c128_i32_54 : i32 to index
        %c1_55 = arith.constant 1 : index
        %124 = arith.muli %c1_55, %123 : index
        %125 = arith.muli %124, %122 : index
        %126 = arith.muli %125, %121 : index
        %127 = arith.muli %126, %c1_51 : index
        %128 = pto.make_tensor_view %arg7, shape = [%c1_51, %c1_51, %121, %122, %123] strides = [%127, %126, %125, %124, %c1_55] : !pto.tensor_view<?x?x?x?x?xf32>
        %c0_56 = arith.constant 0 : index
        %129 = arith.index_cast %118 : i32 to index
        %c1_57 = arith.constant 1 : index
        %c16_58 = arith.constant 16 : index
        %c128_59 = arith.constant 128 : index
        %130 = pto.partition_view %128, offsets = [%c0_56, %c0_56, %c0_56, %c0_56, %129], sizes = [%c1_57, %c1_57, %c1_57, %c16_58, %c128_59] : !pto.tensor_view<?x?x?x?x?xf32> -> !pto.partition_tensor_view<16x128xf32>
        %131 = pto.tstore ins(%6 : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=128, v_row=16, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) outs(%130 : !pto.partition_tensor_view<16x128xf32>) -> tensor<16x128xf32>
        pto.barrier <PIPE_ALL>
        pto.sync.set <PIPE_FIX>, 3
        pto.sync.set <PIPE_FIX>, 19
        pto.sync.wait <PIPE_MTE1>, 4
        pto.sync.wait <PIPE_MTE1>, 20
      }
      pto.sync.wait <PIPE_MTE1>, 8
      pto.sync.wait <PIPE_MTE1>, 24
    }
    pto.section.vector {
      %cst = arith.constant 0.000000e+00 : f32
      pto.texpands ins(%cst : f32) outs(%7 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      %cst_0 = arith.constant 0.000000e+00 : f32
      pto.texpands ins(%cst_0 : f32) outs(%8 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      %cst_1 = arith.constant -1.07374182E+9 : f32
      pto.texpands ins(%cst_1 : f32) outs(%9 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.barrier <PIPE_ALL>
      %c0_i32 = arith.constant 0 : i32
      %22 = arith.index_cast %c0_i32 : i32 to index
      %c32_i32 = arith.constant 32 : i32
      %23 = arith.index_cast %c32_i32 : i32 to index
      %c1 = arith.constant 1 : index
      %24 = arith.addi %22, %23 : index
      scf.for %arg11 = %22 to %24 step %c1 {
        %69 = arith.index_cast %arg11 : index to i32
        %70 = arith.divsi %1, %arg9 : i32
        %71 = arith.remsi %70, %arg8 : i32
        %72 = arith.muli %71, %arg9 : i32
        %c8192_i32_18 = arith.constant 8192 : i32
        %73 = arith.muli %72, %c8192_i32_18 : i32
        %74 = arith.remsi %1, %arg9 : i32
        %c8192_i32_19 = arith.constant 8192 : i32
        %75 = arith.muli %74, %c8192_i32_19 : i32
        %76 = arith.addi %73, %75 : i32
        %77 = arith.divsi %1, %arg9 : i32
        %78 = arith.divsi %77, %arg8 : i32
        %c4_i32_20 = arith.constant 4 : i32
        %79 = arith.remsi %78, %c4_i32_20 : i32
        %c2048_i32_21 = arith.constant 2048 : i32
        %80 = arith.muli %79, %c2048_i32_21 : i32
        %81 = arith.addi %76, %80 : i32
        %c64_i32_22 = arith.constant 64 : i32
        %82 = arith.muli %69, %c64_i32_22 : i32
        %83 = arith.addi %81, %82 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %84 = arith.muli %arg8, %arg9 : i32
        %c8192_i32_24 = arith.constant 8192 : i32
        %85 = arith.muli %84, %c8192_i32_24 : i32
        %c64_i32_25 = arith.constant 64 : i32
        %86 = arith.muli %69, %c64_i32_25 : i32
        %87 = arith.subi %85, %86 : i32
        %88 = arith.divsi %1, %arg9 : i32
        %89 = arith.divsi %88, %arg8 : i32
        %c4_i32_26 = arith.constant 4 : i32
        %90 = arith.remsi %89, %c4_i32_26 : i32
        %c2048_i32_27 = arith.constant 2048 : i32
        %91 = arith.muli %90, %c2048_i32_27 : i32
        %92 = arith.subi %87, %91 : i32
        %93 = arith.divsi %1, %arg9 : i32
        %94 = arith.remsi %93, %arg8 : i32
        %95 = arith.muli %94, %arg9 : i32
        %c8192_i32_28 = arith.constant 8192 : i32
        %96 = arith.muli %95, %c8192_i32_28 : i32
        %97 = arith.subi %92, %96 : i32
        %98 = arith.remsi %1, %arg9 : i32
        %c8192_i32_29 = arith.constant 8192 : i32
        %99 = arith.muli %98, %c8192_i32_29 : i32
        %100 = arith.subi %97, %99 : i32
        %c64_i32_30 = arith.constant 64 : i32
        %c1_31 = arith.constant 1 : index
        %101 = arith.index_cast %arg8 : i32 to index
        %102 = arith.index_cast %arg9 : i32 to index
        %c4_i32_32 = arith.constant 4 : i32
        %103 = arith.index_cast %c4_i32_32 : i32 to index
        %c2048_i32_33 = arith.constant 2048 : i32
        %104 = arith.index_cast %c2048_i32_33 : i32 to index
        %c1_34 = arith.constant 1 : index
        %105 = arith.muli %c1_34, %104 : index
        %106 = arith.muli %105, %103 : index
        %107 = arith.muli %106, %102 : index
        %108 = arith.muli %107, %101 : index
        %109 = pto.make_tensor_view %arg2, shape = [%c1_31, %101, %102, %103, %104] strides = [%108, %107, %106, %105, %c1_34] : !pto.tensor_view<?x?x?x?x?xi32>
        %c0_35 = arith.constant 0 : index
        %110 = arith.index_cast %83 : i32 to index
        %c1_36 = arith.constant 1 : index
        %c1_37 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %111 = pto.partition_view %109, offsets = [%c0_35, %c0_35, %c0_35, %c0_35, %110], sizes = [%c1_36, %c1_36, %c1_36, %c1_37, %c64] : !pto.tensor_view<?x?x?x?x?xi32> -> !pto.partition_tensor_view<1x64xi32>
        %112 = pto.tload ins(%111 : !pto.partition_tensor_view<1x64xi32>) outs(%10 : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> tensor<1x64xi32>
        pto.barrier <PIPE_ALL>
        %c0_i32_38 = arith.constant 0 : i32
        %113 = arith.index_cast %c0_i32_38 : i32 to index
        %c32_i32_39 = arith.constant 32 : i32
        %114 = arith.index_cast %c32_i32_39 : i32 to index
        %c1_40 = arith.constant 1 : index
        %115 = arith.addi %113, %114 : index
        scf.for %arg12 = %113 to %115 step %c1_40 {
          %179 = arith.index_cast %arg12 : index to i32
          %180 = arith.divsi %1, %arg9 : i32
          %181 = arith.remsi %180, %arg8 : i32
          %182 = arith.muli %181, %arg10 : i32
          %c512_i32_96 = arith.constant 512 : i32
          %183 = arith.muli %182, %c512_i32_96 : i32
          %c32_i32_97 = arith.constant 32 : i32
          %184 = arith.muli %21, %c32_i32_97 : i32
          %185 = arith.addi %184, %179 : i32
          %186 = arith.index_cast %185 : i32 to index
          %187 = pto.tgetval ins(%10, %186 : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index) outs : i32
          %c512_i32_98 = arith.constant 512 : i32
          %188 = arith.muli %187, %c512_i32_98 : i32
          %189 = arith.addi %183, %188 : i32
          %190 = arith.divsi %1, %arg9 : i32
          %191 = arith.divsi %190, %arg8 : i32
          %c4_i32_99 = arith.constant 4 : i32
          %192 = arith.remsi %191, %c4_i32_99 : i32
          %c128_i32_100 = arith.constant 128 : i32
          %193 = arith.muli %192, %c128_i32_100 : i32
          %194 = arith.addi %189, %193 : i32
          %c0_i32_101 = arith.constant 0 : i32
          %195 = arith.muli %arg8, %arg10 : i32
          %c512_i32_102 = arith.constant 512 : i32
          %196 = arith.muli %195, %c512_i32_102 : i32
          %197 = arith.divsi %1, %arg9 : i32
          %198 = arith.divsi %197, %arg8 : i32
          %c4_i32_103 = arith.constant 4 : i32
          %199 = arith.remsi %198, %c4_i32_103 : i32
          %c128_i32_104 = arith.constant 128 : i32
          %200 = arith.muli %199, %c128_i32_104 : i32
          %201 = arith.subi %196, %200 : i32
          %202 = arith.divsi %1, %arg9 : i32
          %203 = arith.remsi %202, %arg8 : i32
          %204 = arith.muli %203, %arg10 : i32
          %c512_i32_105 = arith.constant 512 : i32
          %205 = arith.muli %204, %c512_i32_105 : i32
          %206 = arith.subi %201, %205 : i32
          %c32_i32_106 = arith.constant 32 : i32
          %207 = arith.muli %21, %c32_i32_106 : i32
          %208 = arith.addi %207, %179 : i32
          %209 = arith.index_cast %208 : i32 to index
          %210 = pto.tgetval ins(%10, %209 : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index) outs : i32
          %c512_i32_107 = arith.constant 512 : i32
          %211 = arith.muli %210, %c512_i32_107 : i32
          %212 = arith.subi %206, %211 : i32
          %c128_i32_108 = arith.constant 128 : i32
          %c1_109 = arith.constant 1 : index
          %213 = arith.index_cast %arg8 : i32 to index
          %214 = arith.index_cast %arg10 : i32 to index
          %c4_i32_110 = arith.constant 4 : i32
          %215 = arith.index_cast %c4_i32_110 : i32 to index
          %c128_i32_111 = arith.constant 128 : i32
          %216 = arith.index_cast %c128_i32_111 : i32 to index
          %c1_112 = arith.constant 1 : index
          %217 = arith.muli %c1_112, %216 : index
          %218 = arith.muli %217, %215 : index
          %219 = arith.muli %218, %214 : index
          %220 = arith.muli %219, %213 : index
          %221 = pto.make_tensor_view %arg1, shape = [%c1_109, %213, %214, %215, %216] strides = [%220, %219, %218, %217, %c1_112] : !pto.tensor_view<?x?x?x?x?xf16>
          %c0_113 = arith.constant 0 : index
          %222 = arith.index_cast %194 : i32 to index
          %c1_114 = arith.constant 1 : index
          %c1_115 = arith.constant 1 : index
          %c128_116 = arith.constant 128 : index
          %223 = pto.partition_view %221, offsets = [%c0_113, %c0_113, %c0_113, %c0_113, %222], sizes = [%c1_114, %c1_114, %c1_114, %c1_115, %c128_116] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<1x128xf16>
          %224 = pto.tload ins(%223 : !pto.partition_tensor_view<1x128xf16>) outs(%11 : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> tensor<1x128xf16>
          pto.barrier <PIPE_ALL>
          %c0_i32_117 = arith.constant 0 : i32
          %c8192_i32_118 = arith.constant 8192 : i32
          %225 = arith.muli %1, %c8192_i32_118 : i32
          %c4096_i32 = arith.constant 4096 : i32
          %226 = arith.muli %21, %c4096_i32 : i32
          %227 = arith.addi %225, %226 : i32
          %c128_i32_119 = arith.constant 128 : i32
          %228 = arith.muli %179, %c128_i32_119 : i32
          %229 = arith.addi %227, %228 : i32
          %c128_i32_120 = arith.constant 128 : i32
          %c17891328_i32 = arith.constant 17891328 : i32
          %c128_i32_121 = arith.constant 128 : i32
          %230 = arith.muli %179, %c128_i32_121 : i32
          %231 = arith.subi %c17891328_i32, %230 : i32
          %c4096_i32_122 = arith.constant 4096 : i32
          %232 = arith.muli %21, %c4096_i32_122 : i32
          %233 = arith.subi %231, %232 : i32
          %c8192_i32_123 = arith.constant 8192 : i32
          %234 = arith.muli %1, %c8192_i32_123 : i32
          %235 = arith.subi %233, %234 : i32
          %c1_124 = arith.constant 1 : index
          %c2184_i32_125 = arith.constant 2184 : i32
          %236 = arith.index_cast %c2184_i32_125 : i32 to index
          %c64_i32_126 = arith.constant 64 : i32
          %237 = arith.index_cast %c64_i32_126 : i32 to index
          %c128_i32_127 = arith.constant 128 : i32
          %238 = arith.index_cast %c128_i32_127 : i32 to index
          %c1_128 = arith.constant 1 : index
          %239 = arith.muli %c1_128, %238 : index
          %240 = arith.muli %239, %237 : index
          %241 = arith.muli %240, %236 : index
          %242 = arith.muli %241, %c1_124 : index
          %243 = pto.make_tensor_view %arg4, shape = [%c1_124, %c1_124, %236, %237, %238] strides = [%242, %241, %240, %239, %c1_128] : !pto.tensor_view<?x?x?x?x?xf16>
          %c0_129 = arith.constant 0 : index
          %244 = arith.index_cast %229 : i32 to index
          %c1_130 = arith.constant 1 : index
          %c1_131 = arith.constant 1 : index
          %c128_132 = arith.constant 128 : index
          %245 = pto.partition_view %243, offsets = [%c0_129, %c0_129, %c0_129, %c0_129, %244], sizes = [%c1_130, %c1_130, %c1_130, %c1_131, %c128_132] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<1x128xf16>
          %246 = pto.tstore ins(%11 : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%245 : !pto.partition_tensor_view<1x128xf16>) -> tensor<1x128xf16>
          pto.barrier <PIPE_ALL>
        }
        pto.sync.set <PIPE_MTE3>, 0
        %cst_41 = arith.constant 0.000000e+00 : f32
        pto.texpands ins(%cst_41 : f32) outs(%12 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.barrier <PIPE_ALL>
        %c0_i32_42 = arith.constant 0 : i32
        %c0_i32_43 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c8_i32_44 = arith.constant 8 : i32
        %116 = pto.tmov ins(%9 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> tensor<1x8xf32>
        pto.barrier <PIPE_ALL>
        pto.sync.wait <PIPE_V>, 1
        %c1024_i32_45 = arith.constant 1024 : i32
        %117 = arith.muli %1, %c1024_i32_45 : i32
        %c512_i32 = arith.constant 512 : i32
        %118 = arith.muli %21, %c512_i32 : i32
        %119 = arith.addi %117, %118 : i32
        %c0_i32_46 = arith.constant 0 : i32
        %c2236416_i32 = arith.constant 2236416 : i32
        %c512_i32_47 = arith.constant 512 : i32
        %120 = arith.muli %21, %c512_i32_47 : i32
        %121 = arith.subi %c2236416_i32, %120 : i32
        %c1024_i32_48 = arith.constant 1024 : i32
        %122 = arith.muli %1, %c1024_i32_48 : i32
        %123 = arith.subi %121, %122 : i32
        %c512_i32_49 = arith.constant 512 : i32
        %c1_50 = arith.constant 1 : index
        %c2184_i32 = arith.constant 2184 : i32
        %124 = arith.index_cast %c2184_i32 : i32 to index
        %c16_i32 = arith.constant 16 : i32
        %125 = arith.index_cast %c16_i32 : i32 to index
        %c64_i32_51 = arith.constant 64 : i32
        %126 = arith.index_cast %c64_i32_51 : i32 to index
        %c1_52 = arith.constant 1 : index
        %127 = arith.muli %c1_52, %126 : index
        %128 = arith.muli %127, %125 : index
        %129 = arith.muli %128, %124 : index
        %130 = arith.muli %129, %c1_50 : index
        %131 = pto.make_tensor_view %arg5, shape = [%c1_50, %c1_50, %124, %125, %126] strides = [%130, %129, %128, %127, %c1_52] : !pto.tensor_view<?x?x?x?x?xf32>
        %c0_53 = arith.constant 0 : index
        %132 = arith.index_cast %119 : i32 to index
        %c1_54 = arith.constant 1 : index
        %c8_55 = arith.constant 8 : index
        %c64_56 = arith.constant 64 : index
        %133 = pto.partition_view %131, offsets = [%c0_53, %c0_53, %c0_53, %c0_53, %132], sizes = [%c1_54, %c1_54, %c1_54, %c8_55, %c64_56] : !pto.tensor_view<?x?x?x?x?xf32> -> !pto.partition_tensor_view<8x64xf32>
        %134 = pto.tload ins(%133 : !pto.partition_tensor_view<8x64xf32>) outs(%14 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> tensor<8x64xf32>
        pto.barrier <PIPE_ALL>
        pto.tadd ins(%12, %14 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%12 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %cst_57 = arith.constant 0.0883883461 : f32
        pto.tmuls ins(%12, %cst_57 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%12 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %135 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
        %136 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
        pto.treshape ins(%9 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%136 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
        pto.trowmax ins(%12, %135 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%136 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
        pto.treshape ins(%136 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%9 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.tmax ins(%9, %13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%9 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.tsub ins(%13, %9 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.texp ins(%13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %c0_i32_58 = arith.constant 0 : i32
        %137 = arith.index_cast %c0_i32_58 : i32 to index
        %c8_i32_59 = arith.constant 8 : i32
        %138 = arith.index_cast %c8_i32_59 : i32 to index
        %c1_60 = arith.constant 1 : index
        %139 = arith.addi %137, %138 : index
        scf.for %arg12 = %137 to %139 step %c1_60 {
          %179 = arith.index_cast %arg12 : index to i32
          %c0_96 = arith.constant 0 : index
          %c64_i32_97 = arith.constant 64 : i32
          %180 = arith.muli %179, %c64_i32_97 : i32
          %181 = arith.index_cast %180 : i32 to index
          %182 = pto.subset %12[%c0_96, %181] sizes [8, 64] : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
          %c0_98 = arith.constant 0 : index
          %c64_i32_99 = arith.constant 64 : i32
          %183 = arith.muli %179, %c64_i32_99 : i32
          %184 = arith.index_cast %183 : i32 to index
          %185 = pto.subset %12[%c0_98, %184] sizes [8, 64] : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
          %186 = arith.index_cast %179 : i32 to index
          %187 = pto.tgetval ins(%9, %186 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index) outs : f32
          pto.tsubs ins(%185, %187 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%182 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        }
        pto.texp ins(%12 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%12 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %140 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
        %141 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
        pto.treshape ins(%16 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%141 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
        pto.trowsum ins(%12, %140 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%141 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
        pto.treshape ins(%141 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%16 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.tmul ins(%8, %13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%8 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.tadd ins(%8, %16 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%8 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %142 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
        pto.treshape ins(%13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%142 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
        pto.trowexpandmul ins(%7, %142 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%7 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.treshape ins(%142 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%13 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %c0_i32_61 = arith.constant 0 : i32
        %c0_i32_62 = arith.constant 0 : i32
        %c512_i32_63 = arith.constant 512 : i32
        %c512_i32_64 = arith.constant 512 : i32
        pto.tcvt ins(%12 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%17 : !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.barrier <PIPE_ALL>
        %c0_i32_65 = arith.constant 0 : i32
        %c1024_i32_66 = arith.constant 1024 : i32
        %143 = arith.muli %1, %c1024_i32_66 : i32
        %c512_i32_67 = arith.constant 512 : i32
        %144 = arith.muli %21, %c512_i32_67 : i32
        %145 = arith.addi %143, %144 : i32
        %c512_i32_68 = arith.constant 512 : i32
        %c2236416_i32_69 = arith.constant 2236416 : i32
        %c512_i32_70 = arith.constant 512 : i32
        %146 = arith.muli %21, %c512_i32_70 : i32
        %147 = arith.subi %c2236416_i32_69, %146 : i32
        %c1024_i32_71 = arith.constant 1024 : i32
        %148 = arith.muli %1, %c1024_i32_71 : i32
        %149 = arith.subi %147, %148 : i32
        %c1_72 = arith.constant 1 : index
        %c2184_i32_73 = arith.constant 2184 : i32
        %150 = arith.index_cast %c2184_i32_73 : i32 to index
        %c16_i32_74 = arith.constant 16 : i32
        %151 = arith.index_cast %c16_i32_74 : i32 to index
        %c64_i32_75 = arith.constant 64 : i32
        %152 = arith.index_cast %c64_i32_75 : i32 to index
        %c1_76 = arith.constant 1 : index
        %153 = arith.muli %c1_76, %152 : index
        %154 = arith.muli %153, %151 : index
        %155 = arith.muli %154, %150 : index
        %156 = arith.muli %155, %c1_72 : index
        %157 = pto.make_tensor_view %arg6, shape = [%c1_72, %c1_72, %150, %151, %152] strides = [%156, %155, %154, %153, %c1_76] : !pto.tensor_view<?x?x?x?x?xf16>
        %c0_77 = arith.constant 0 : index
        %158 = arith.index_cast %145 : i32 to index
        %c1_78 = arith.constant 1 : index
        %c8_79 = arith.constant 8 : index
        %c64_80 = arith.constant 64 : index
        %159 = pto.partition_view %157, offsets = [%c0_77, %c0_77, %c0_77, %c0_77, %158], sizes = [%c1_78, %c1_78, %c1_78, %c8_79, %c64_80] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<8x64xf16>
        %160 = pto.tstore ins(%17 : !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%159 : !pto.partition_tensor_view<8x64xf16>) -> tensor<8x64xf16>
        pto.barrier <PIPE_ALL>
        pto.sync.set <PIPE_MTE3>, 2
        pto.sync.wait <PIPE_V>, 3
        pto.barrier <PIPE_ALL>
        %c2048_i32_81 = arith.constant 2048 : i32
        %161 = arith.muli %1, %c2048_i32_81 : i32
        %c1024_i32_82 = arith.constant 1024 : i32
        %162 = arith.muli %21, %c1024_i32_82 : i32
        %163 = arith.addi %161, %162 : i32
        %c0_i32_83 = arith.constant 0 : i32
        %c4472832_i32 = arith.constant 4472832 : i32
        %c1024_i32_84 = arith.constant 1024 : i32
        %164 = arith.muli %21, %c1024_i32_84 : i32
        %165 = arith.subi %c4472832_i32, %164 : i32
        %c2048_i32_85 = arith.constant 2048 : i32
        %166 = arith.muli %1, %c2048_i32_85 : i32
        %167 = arith.subi %165, %166 : i32
        %c1024_i32_86 = arith.constant 1024 : i32
        %c1_87 = arith.constant 1 : index
        %c2184_i32_88 = arith.constant 2184 : i32
        %168 = arith.index_cast %c2184_i32_88 : i32 to index
        %c16_i32_89 = arith.constant 16 : i32
        %169 = arith.index_cast %c16_i32_89 : i32 to index
        %c128_i32_90 = arith.constant 128 : i32
        %170 = arith.index_cast %c128_i32_90 : i32 to index
        %c1_91 = arith.constant 1 : index
        %171 = arith.muli %c1_91, %170 : index
        %172 = arith.muli %171, %169 : index
        %173 = arith.muli %172, %168 : index
        %174 = arith.muli %173, %c1_87 : index
        %175 = pto.make_tensor_view %arg7, shape = [%c1_87, %c1_87, %168, %169, %170] strides = [%174, %173, %172, %171, %c1_91] : !pto.tensor_view<?x?x?x?x?xf32>
        %c0_92 = arith.constant 0 : index
        %176 = arith.index_cast %163 : i32 to index
        %c1_93 = arith.constant 1 : index
        %c8_94 = arith.constant 8 : index
        %c128_95 = arith.constant 128 : index
        %177 = pto.partition_view %175, offsets = [%c0_92, %c0_92, %c0_92, %c0_92, %176], sizes = [%c1_93, %c1_93, %c1_93, %c8_94, %c128_95] : !pto.tensor_view<?x?x?x?x?xf32> -> !pto.partition_tensor_view<8x128xf32>
        %178 = pto.tload ins(%177 : !pto.partition_tensor_view<8x128xf32>) outs(%18 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> tensor<8x128xf32>
        pto.barrier <PIPE_ALL>
        pto.tadd ins(%7, %18 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%7 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        pto.barrier <PIPE_ALL>
        pto.sync.set <PIPE_V>, 4
        pto.barrier <PIPE_ALL>
      }
      %25 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
      pto.treshape ins(%8 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%25 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
      pto.trowexpanddiv ins(%7, %25 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%7 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.treshape ins(%25 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=8, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%8 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      %c0_i32_2 = arith.constant 0 : i32
      %c0_i32_3 = arith.constant 0 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %c1024_i32_4 = arith.constant 1024 : i32
      pto.tcvt ins(%7 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%19 : !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.barrier <PIPE_ALL>
      %c0_i32_5 = arith.constant 0 : i32
      %26 = arith.divsi %1, %arg9 : i32
      %27 = arith.remsi %26, %arg8 : i32
      %28 = arith.muli %27, %arg9 : i32
      %c8192_i32 = arith.constant 8192 : i32
      %29 = arith.muli %28, %c8192_i32 : i32
      %30 = arith.remsi %1, %arg9 : i32
      %c8192_i32_6 = arith.constant 8192 : i32
      %31 = arith.muli %30, %c8192_i32_6 : i32
      %32 = arith.addi %29, %31 : i32
      %33 = arith.divsi %1, %arg9 : i32
      %34 = arith.divsi %33, %arg8 : i32
      %c4_i32 = arith.constant 4 : i32
      %35 = arith.remsi %34, %c4_i32 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %36 = arith.muli %35, %c2048_i32 : i32
      %37 = arith.addi %32, %36 : i32
      %c1024_i32_7 = arith.constant 1024 : i32
      %38 = arith.muli %21, %c1024_i32_7 : i32
      %39 = arith.addi %37, %38 : i32
      %c1024_i32_8 = arith.constant 1024 : i32
      %40 = arith.muli %arg8, %arg9 : i32
      %c8192_i32_9 = arith.constant 8192 : i32
      %41 = arith.muli %40, %c8192_i32_9 : i32
      %c1024_i32_10 = arith.constant 1024 : i32
      %42 = arith.muli %21, %c1024_i32_10 : i32
      %43 = arith.subi %41, %42 : i32
      %44 = arith.divsi %1, %arg9 : i32
      %45 = arith.divsi %44, %arg8 : i32
      %c4_i32_11 = arith.constant 4 : i32
      %46 = arith.remsi %45, %c4_i32_11 : i32
      %c2048_i32_12 = arith.constant 2048 : i32
      %47 = arith.muli %46, %c2048_i32_12 : i32
      %48 = arith.subi %43, %47 : i32
      %49 = arith.divsi %1, %arg9 : i32
      %50 = arith.remsi %49, %arg8 : i32
      %51 = arith.muli %50, %arg9 : i32
      %c8192_i32_13 = arith.constant 8192 : i32
      %52 = arith.muli %51, %c8192_i32_13 : i32
      %53 = arith.subi %48, %52 : i32
      %54 = arith.remsi %1, %arg9 : i32
      %c8192_i32_14 = arith.constant 8192 : i32
      %55 = arith.muli %54, %c8192_i32_14 : i32
      %56 = arith.subi %53, %55 : i32
      %c1_15 = arith.constant 1 : index
      %57 = arith.index_cast %arg8 : i32 to index
      %58 = arith.index_cast %arg9 : i32 to index
      %c64_i32 = arith.constant 64 : i32
      %59 = arith.index_cast %c64_i32 : i32 to index
      %c128_i32 = arith.constant 128 : i32
      %60 = arith.index_cast %c128_i32 : i32 to index
      %c1_16 = arith.constant 1 : index
      %61 = arith.muli %c1_16, %60 : index
      %62 = arith.muli %61, %59 : index
      %63 = arith.muli %62, %58 : index
      %64 = arith.muli %63, %57 : index
      %65 = pto.make_tensor_view %arg3, shape = [%c1_15, %57, %58, %59, %60] strides = [%64, %63, %62, %61, %c1_16] : !pto.tensor_view<?x?x?x?x?xf16>
      %c0 = arith.constant 0 : index
      %66 = arith.index_cast %39 : i32 to index
      %c1_17 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %67 = pto.partition_view %65, offsets = [%c0, %c0, %c0, %c0, %66], sizes = [%c1_17, %c1_17, %c1_17, %c8, %c128] : !pto.tensor_view<?x?x?x?x?xf16> -> !pto.partition_tensor_view<8x128xf16>
      %68 = pto.tstore ins(%19 : !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%67 : !pto.partition_tensor_view<8x128xf16>) -> tensor<8x128xf16>
      pto.barrier <PIPE_ALL>
      pto.sync.set <PIPE_MTE3>, 8
    }
    return
  }
}
"""

if __name__ == "__main__":
    print(build())
