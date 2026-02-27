"""
Dynamic-offset/dynamic-size partition_view -> tload -> tstore.
Text is emitted as hand-crafted MLIR assembly to keep output clean (non-generic).
Offsets:  %arg2..%arg6
Sizes:    %arg7..%arg11
valid_row = arg7*arg8*arg9*arg10
valid_col = arg11
Tile capacity fixed 256x16; valid dims are dynamic for tail blocks.
"""

MLIR_TEXT = r'''module {
  func.func @run_partition(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                           %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index,
                           %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c64_1 = arith.constant 64 : index
    %c65536 = arith.constant 65536 : index
    %c65536_2 = arith.constant 65536 : index
    %c4096 = arith.constant 4096 : index
    %c64_2 = arith.constant 64 : index
    %c1_5 = arith.constant 1 : index

    %base = pto.make_tensor_view %arg0, shape = [%c1, %c1_0, %c16, %c64, %c64_1] strides = [%c65536, %c65536_2, %c4096, %c64_2, %c1_5]
           : !pto.tensor_view<1x1x16x64x64xf32>

    %part = pto.partition_view %base,
             offsets = [%arg2, %arg3, %arg4, %arg5, %arg6],
             sizes   = [%arg7, %arg8, %arg9, %arg10, %arg11]
           : !pto.tensor_view<1x1x16x64x64xf32> -> !pto.partition_tensor_view<?x?x?x?x?xf32>

    %tmp0 = arith.muli %arg7, %arg8 : index
    %tmp1 = arith.muli %tmp0, %arg9 : index
    %tmp2 = arith.muli %tmp1, %arg10 : index

    %tile = pto.alloc_tile valid_row = %tmp2 valid_col = %arg11
           : !pto.tile_buf<loc=vec, dtype=f32, rows=256, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%part : !pto.partition_tensor_view<?x?x?x?x?xf32>)
            outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=256, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    %dst_view = pto.make_tensor_view %arg1, shape = [%c1, %c1_0, %c16, %c64, %c64_1] strides = [%c65536, %c65536_2, %c4096, %c64_2, %c1_5]
               : !pto.tensor_view<1x1x16x64x64xf32>

    %dst_part = pto.partition_view %dst_view,
                offsets = [%arg2, %arg3, %arg4, %arg5, %arg6],
                sizes   = [%arg7, %arg8, %arg9, %arg10, %arg11]
               : !pto.tensor_view<1x1x16x64x64xf32> -> !pto.partition_tensor_view<?x?x?x?x?xf32>

    pto.tstore ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=256, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
             outs(%dst_part : !pto.partition_tensor_view<?x?x?x?x?xf32>)
    return
  }
}
'''

if __name__ == "__main__":
    print(MLIR_TEXT)
