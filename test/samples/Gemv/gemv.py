from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)
            ptr_f16 = pto.PtrType.get(f16, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            # TGEMV: A(1xK) * B(KxN) -> C(1xN)
            M = 1
            M_ALIGN = 16
            K = 256
            N = 32

            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_a = pto.PartitionTensorViewType.get([M, K], f16, ctx)
            tile_view_b = pto.PartitionTensorViewType.get([K, N], f16, ctx)
            tile_view_c = pto.PartitionTensorViewType.get([M, N], f32, ctx)

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)

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

            tile_buf_a_mat = pto.TileBufType.get([M, K], f16, mat, [M, K], cfg_a_mat, ctx)
            tile_buf_b_mat = pto.TileBufType.get([K, N], f16, mat, [K, N], cfg_b_mat, ctx)
            tile_buf_a = pto.TileBufType.get([M, K], f16, left, [M, K], cfg_left, ctx)
            tile_buf_b = pto.TileBufType.get([K, N], f16, right, [K, N], cfg_right, ctx)
            # TGEMV expects the ACC tile rows to be aligned to 16 on some SoCs.
            tile_buf_c = pto.TileBufType.get([M_ALIGN, N], f32, acc, [M, N], cfg_acc, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("gemv_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cM = arith.ConstantOp(IndexType.get(ctx), M).result
                cK = arith.ConstantOp(IndexType.get(ctx), K).result
                cN = arith.ConstantOp(IndexType.get(ctx), N).result

                arg_a, arg_b, arg_out = entry.arguments

                tv_a = pto.MakeTensorViewOp(tv2_f16, arg_a, [cM, cK], [cK, c1]).result
                tv_b = pto.MakeTensorViewOp(tv2_f16, arg_b, [cK, cN], [cN, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [cM, cN], [cN, c1]).result

                sv_a = pto.PartitionViewOp(tile_view_a, tv_a, offsets=[c0, c0], sizes=[cM, cK]).result
                sv_b = pto.PartitionViewOp(tile_view_b, tv_b, offsets=[c0, c0], sizes=[cK, cN]).result

                a_mat = pto.AllocTileOp(tile_buf_a_mat).result
                b_mat = pto.AllocTileOp(tile_buf_b_mat).result
                a_tile = pto.AllocTileOp(tile_buf_a).result
                b_tile = pto.AllocTileOp(tile_buf_b).result
                c_tile = pto.AllocTileOp(tile_buf_c).result

                pto.TLoadOp(None, sv_a, a_mat)
                pto.TLoadOp(None, sv_b, b_mat)

                # A: TEXTRACT (Mat -> Left), B: TMOV (Mat -> Right)
                pto.TExtractOp(a_mat, c0, c0, a_tile)
                pto.TMovOp(None, b_mat, b_tile)

                pto.TGemvOp(None, a_tile, b_tile, c_tile)

                sv_out = pto.PartitionViewOp(tile_view_c, tv_out, offsets=[c0, c0], sizes=[cM, cN]).result
                pto.TStoreOp(None, c_tile, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
