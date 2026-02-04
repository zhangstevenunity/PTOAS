from mlir.ir import Context, Location, Module, InsertionPoint, IntegerType
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType

def build_mixed_shape():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)  # 用于传递动态维度的参数
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)

            # Configs
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)

            # [核心定义 1] 混合类型的 TileBufType
            # valid_shape=[-1, 32] 表示: Row 是动态的(?), Col 是静态的(32)
            # 对应 IR 类型: !pto.tile_buf<..., v_row=?, v_col=32 ...>
            tile_buf_mixed = pto.TileBufType.get([32, 32], f32, vec, [-1, 32], cfg, ctx)

            # 全静态 TileBufType (用于对比)
            tile_buf_static = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            # [核心定义 2] 函数签名
            # 只需要 1 个 i32 参数，因为只有一个维度是动态的
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, i32], [])

            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_add_kernel_mixed_shape", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                scale = arith.ConstantOp(f32, 3.14).result

                # 获取参数: 只获取一个动态维度参数
                arg0, arg1, arg2, arg_vrow_i32 = entry.arguments

                # 类型转换: i32 -> index
                v_row_idx = arith.IndexCastOp(IndexType.get(ctx), arg_vrow_i32).result

                # 构造 Tensor Views
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c32, c32], [c32, c1]).result

                # Subviews
                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result

                # [修改] AllocTileOp 使用动态参数
                # 对应 MLIR: %5 = pto.alloc_tile valid_row=%v_row valid_col=%v_col : <..., v_row=?, v_col=32>
                tb0 = pto.AllocTileOp(tile_buf_mixed, valid_row=v_row_idx, valid_col=None).result

                # 其他静态 Tile 保持不变
                tb1 = pto.AllocTileOp(tile_buf_static).result
                tb2 = pto.AllocTileOp(tile_buf_static).result

                # Load / Compute / Store
                # 注意 tb0 的类型是动态的，所以 ins/outs 类型要匹配
                pto.TLoadOp(None, sv0, tb0)
                pto.TLoadOp(None, sv1, tb1)

                pto.TAddOp(tb0, tb1, tb2)

                sv2 = pto.PartitionViewOp(tile_view_32, tv2, offsets=[c0, c0], sizes=[c32, c32]).result
                pto.TStoreOp(None, tb2, sv2)

                func.ReturnOp([])

            m.operation.verify()
            return m

if __name__ == "__main__":
    print(build_mixed_shape())