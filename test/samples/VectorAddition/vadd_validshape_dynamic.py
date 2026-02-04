from mlir.ir import Context, Location, Module, InsertionPoint, IntegerType
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)  # 定义 i32 类型
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            
            # [修改] 定义全动态 valid shape 的 TileBufType: valid_shape=[-1, -1] 对应 MLIR 中的 <..., v_row=?, v_col=?>
            tile_buf_dynamic = pto.TileBufType.get([32, 32], f32, vec, [-1, -1], cfg, ctx)
            
            # 静态 TileBufType (用于对比)
            tile_buf_static = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            # [修改] 函数签名增加两个 i32 参数: (ptr, ptr, ptr, i32, i32) -> ()
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, i32, i32], [])
            
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_add_kernel_2d_dynamic", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                scale = arith.ConstantOp(f32, 3.14).result

                # [修改] 获取参数
                arg0, arg1, arg2, arg_vrow_i32, arg_vcol_i32 = entry.arguments

                # [修改] 将 i32 参数转换为 index 类型
                # 对应 MLIR: %v_row = arith.index_cast %arg3 : i32 to index
                v_row_idx = arith.IndexCastOp(IndexType.get(ctx), arg_vrow_i32).result
                v_col_idx = arith.IndexCastOp(IndexType.get(ctx), arg_vcol_i32).result

                # 构造 Tensor Views
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c32, c32], [c32, c1]).result

                # Subviews
                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result

                # [修改] AllocTileOp 使用动态参数
                # 对应 MLIR: %5 = pto.alloc_tile valid_row=%v_row valid_col=%v_col : <..., v_row=?, v_col=?>
                tb0 = pto.AllocTileOp(tile_buf_dynamic, valid_row=v_row_idx, valid_col=v_col_idx).result
                
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
    print(build())