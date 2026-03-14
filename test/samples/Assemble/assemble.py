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
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_f16 = pto.PtrType.get(f16, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)

            tile_view_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            tile_view_f16 = pto.PartitionTensorViewType.get([32, 32], f16, ctx)

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg_mat_f32 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_left_f32 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_right_f32 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_acc_f32 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                1024,
                pd,
                ctx,
            )
            cfg_mat_f16 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_left_f16 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_right_f16 = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )

            a_mat_ty = pto.TileBufType.get([32, 32], f32, mat, [32, 32], cfg_mat_f32, ctx)
            b_mat_ty = pto.TileBufType.get([32, 32], f32, mat, [32, 32], cfg_mat_f32, ctx)
            a_left_ty = pto.TileBufType.get([32, 32], f32, left, [32, 32], cfg_left_f32, ctx)
            b_right_ty = pto.TileBufType.get([32, 32], f32, right, [32, 32], cfg_right_f32, ctx)
            src_acc_ty = pto.TileBufType.get([32, 32], f32, acc, [32, 32], cfg_acc_f32, ctx)
            dst_mat_ty = pto.TileBufType.get([32, 32], f16, mat, [32, 32], cfg_mat_f16, ctx)
            out_left_ty = pto.TileBufType.get([32, 32], f16, left, [32, 32], cfg_left_f16, ctx)
            i_mat_ty = pto.TileBufType.get([32, 32], f16, mat, [32, 32], cfg_mat_f16, ctx)
            i_right_ty = pto.TileBufType.get([32, 32], f16, right, [32, 32], cfg_right_f16, ctx)
            out_acc_ty = pto.TileBufType.get([32, 32], f32, acc, [32, 32], cfg_acc_f32, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f16, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("assemble_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c32 = arith.ConstantOp(idx, 32).result

                arg_a, arg_b, arg_i, arg_out = entry.arguments

                tv_a = pto.MakeTensorViewOp(tv2_f32, arg_a, [c32, c32], [c32, c1]).result
                tv_b = pto.MakeTensorViewOp(tv2_f32, arg_b, [c32, c32], [c32, c1]).result
                tv_i = pto.MakeTensorViewOp(tv2_f16, arg_i, [c32, c32], [c32, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c32, c32], [c32, c1]).result

                sv_a = pto.PartitionViewOp(tile_view_f32, tv_a, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_b = pto.PartitionViewOp(tile_view_f32, tv_b, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_i = pto.PartitionViewOp(tile_view_f16, tv_i, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_out = pto.PartitionViewOp(tile_view_f32, tv_out, offsets=[c0, c0], sizes=[c32, c32]).result

                a_mat = pto.AllocTileOp(a_mat_ty).result
                b_mat = pto.AllocTileOp(b_mat_ty).result
                a_left = pto.AllocTileOp(a_left_ty).result
                b_right = pto.AllocTileOp(b_right_ty).result
                src_acc = pto.AllocTileOp(src_acc_ty).result
                dst_mat = pto.AllocTileOp(dst_mat_ty).result
                out_left = pto.AllocTileOp(out_left_ty).result
                i_mat = pto.AllocTileOp(i_mat_ty).result
                i_right = pto.AllocTileOp(i_right_ty).result
                out_acc = pto.AllocTileOp(out_acc_ty).result

                pto.TLoadOp(None, sv_a, a_mat)
                pto.TLoadOp(None, sv_b, b_mat)
                pto.TMovOp(None, a_mat, a_left)
                pto.TMovOp(None, b_mat, b_right)
                pto.TMatmulOp(None, a_left, b_right, src_acc)

                # Main operation under test: lowering must emit TINSERT(dst, src, row, col).
                pto.TAssembleOp(src_acc, c0, c0, dst_mat)

                pto.TLoadOp(None, sv_i, i_mat)
                pto.TMovOp(None, dst_mat, out_left)
                pto.TMovOp(None, i_mat, i_right)
                pto.TMatmulOp(None, out_left, i_right, out_acc)
                pto.TStoreOp(None, out_acc, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
