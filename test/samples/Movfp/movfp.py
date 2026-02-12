"""
TMOV_FP is implemented in PTO-ISA as a vector-quant move from ACC -> MAT using a
SCALING (fp) tile.

Important PTO-ISA constraints (a2a3/a5):
  - fp tile must live in SCALING (fbuf) and use uint64_t element type.
  - TLOAD cannot load directly into SCALING tiles; load into MAT then TMOV to SCALING.
  - src must be ACC tile; dst must be MAT tile.
"""

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, F32Type, IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            ui64 = IntegerType.get_unsigned(64, ctx)

            ptr_f16 = pto.PtrType.get(f16, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_ui64 = pto.PtrType.get(ui64, ctx)

            # GEMV: A(1xK) * B(KxN) -> ACC(align16 x N)
            M = 1
            M_ALIGN = 16
            K = 256
            N = 32

            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tv2_ui64 = pto.TensorViewType.get(2, ui64, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)

            tile_view_a = pto.PartitionTensorViewType.get([M, K], f16, ctx)
            tile_view_b = pto.PartitionTensorViewType.get([K, N], f16, ctx)
            tile_view_out = pto.PartitionTensorViewType.get([M, N], f32, ctx)
            tile_view_fp = pto.PartitionTensorViewType.get([1, 16], ui64, ctx)

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)

            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg_a_mat = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_b_mat = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_left = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_right = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_acc = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalCSize,
                pd,
                ctx,
            )
            cfg_out_mat = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_fp = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )

            tile_buf_a_mat = pto.TileBufType.get([M, K], f16, mat, [M, K], cfg_a_mat, ctx)
            tile_buf_b_mat = pto.TileBufType.get([K, N], f16, mat, [K, N], cfg_b_mat, ctx)
            tile_buf_a = pto.TileBufType.get([M, K], f16, left, [M, K], cfg_left, ctx)
            tile_buf_b = pto.TileBufType.get([K, N], f16, right, [K, N], cfg_right, ctx)
            tile_buf_acc = pto.TileBufType.get([M_ALIGN, N], f32, acc, [M, N], cfg_acc, ctx)
            tile_buf_out_mat = pto.TileBufType.get([M_ALIGN, N], i8, mat, [M, N], cfg_out_mat, ctx)

            # fp scaling: load into MAT then TMOV -> SCALING
            tile_buf_fp_mat = pto.TileBufType.get([1, 16], ui64, mat, [1, 16], cfg_fp, ctx)
            tile_buf_fp_scaling = pto.TileBufType.get([1, 16], ui64, scaling, [1, 16], cfg_fp, ctx)

            # Function takes 4 arguments: a_ptr, b_ptr, fp_ptr, out_ptr (f32)
            # Note: TMOV_FP produces a MAT (int8) tile; we keep that as an internal
            # value for compilation/runtime coverage but store ACC (f32) to GM so
            # the output is in a plain ND layout and comparable.
            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16, ptr_ui64, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_movfp_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cM = arith.ConstantOp(IndexType.get(ctx), M).result
                cK = arith.ConstantOp(IndexType.get(ctx), K).result
                cN = arith.ConstantOp(IndexType.get(ctx), N).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result

                arg_a, arg_b, arg_fp, arg_out = entry.arguments

                tv_a = pto.MakeTensorViewOp(tv2_f16, arg_a, [cM, cK], [cK, c1]).result
                tv_b = pto.MakeTensorViewOp(tv2_f16, arg_b, [cK, cN], [cN, c1]).result
                tv_fp = pto.MakeTensorViewOp(tv2_ui64, arg_fp, [c1, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [cM, cN], [cN, c1]).result

                sv_a = pto.PartitionViewOp(tile_view_a, tv_a, offsets=[c0, c0], sizes=[cM, cK]).result
                sv_b = pto.PartitionViewOp(tile_view_b, tv_b, offsets=[c0, c0], sizes=[cK, cN]).result
                sv_fp = pto.PartitionViewOp(tile_view_fp, tv_fp, offsets=[c0, c0], sizes=[c1, c16]).result
                sv_out = pto.PartitionViewOp(tile_view_out, tv_out, offsets=[c0, c0], sizes=[cM, cN]).result

                a_mat = pto.AllocTileOp(tile_buf_a_mat).result
                b_mat = pto.AllocTileOp(tile_buf_b_mat).result
                fp_mat = pto.AllocTileOp(tile_buf_fp_mat).result

                a_tile = pto.AllocTileOp(tile_buf_a).result
                b_tile = pto.AllocTileOp(tile_buf_b).result
                fp_scaling = pto.AllocTileOp(tile_buf_fp_scaling).result

                acc_tile = pto.AllocTileOp(tile_buf_acc).result
                out_mat = pto.AllocTileOp(tile_buf_out_mat).result

                pto.TLoadOp(None, sv_a, a_mat)
                pto.TLoadOp(None, sv_b, b_mat)
                pto.TLoadOp(None, sv_fp, fp_mat)

                # Mat -> Left/Right and Mat -> Scaling
                pto.TExtractOp(a_mat, c0, c0, a_tile)
                pto.TMovOp(None, b_mat, b_tile)
                pto.TMovOp(None, fp_mat, fp_scaling)

                # Compute ACC then quantize ACC->MAT using fp (SCALING) tile.
                pto.TGemvOp(None, a_tile, b_tile, acc_tile)
                pto.TMovFPOp(acc_tile, fp_scaling, out_mat)

                pto.TStoreOp(None, acc_tile, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
