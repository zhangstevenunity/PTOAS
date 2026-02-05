from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            # GEMV: matrix-vector multiplication
            # Matrix: M x K, Vector: K x 1, Result: M x 1
            M = 32
            K = 256
            N = 1  # Vector dimension

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_MxK = pto.PartitionTensorViewType.get([M, K], f32, ctx)
            tile_view_Kx1 = pto.PartitionTensorViewType.get([K, N], f32, ctx)
            tile_view_Mx1 = pto.PartitionTensorViewType.get([M, N], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC)

            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_MxK = pto.TileBufType.get([M, K], f32, left, [M, K], cfg, ctx)
            tile_buf_Kx1 = pto.TileBufType.get([K, N], f32, right, [K, N], cfg, ctx)
            tile_buf_Mx1 = pto.TileBufType.get([M, N], f32, acc, [M, N], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("gemv_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cM = arith.ConstantOp(IndexType.get(ctx), M).result
                cK = arith.ConstantOp(IndexType.get(ctx), K).result
                cN = arith.ConstantOp(IndexType.get(ctx), N).result

                arg0, arg1, arg2 = entry.arguments  # matrix, vector, result

                # Create tensor views
                tv_matrix = pto.MakeTensorViewOp(tv2_f32, arg0, [cM, cK], [cK, c1]).result
                # Vector as 2D: K x 1
                tv_vector = pto.MakeTensorViewOp(tv2_f32, arg1, [cK, cN], [cN, c1]).result
                # Result as 2D: M x 1
                tv_result = pto.MakeTensorViewOp(tv2_f32, arg2, [cM, cN], [cN, c1]).result

                # Create partition views
                sv_matrix = pto.PartitionViewOp(tile_view_MxK, tv_matrix, offsets=[c0, c0], sizes=[cM, cK]).result
                sv_vector = pto.PartitionViewOp(tile_view_Kx1, tv_vector, offsets=[c0, c0], sizes=[cK, cN]).result

                # Allocate tiles
                tb_matrix = pto.AllocTileOp(tile_buf_MxK).result
                tb_vector = pto.AllocTileOp(tile_buf_Kx1).result
                tb_result = pto.AllocTileOp(tile_buf_Mx1).result

                # Load data
                pto.TLoadOp(None, sv_matrix, tb_matrix)
                pto.TLoadOp(None, sv_vector, tb_vector)

                # GEMV: matrix-vector multiplication
                pto.TGemvOp(None, tb_matrix, tb_vector, tb_result)

                # Store result
                sv_result = pto.PartitionViewOp(tile_view_Mx1, tv_result, offsets=[c0, c0], sizes=[cM, cN]).result
                pto.TStoreOp(None, tb_result, sv_result)

                func.ReturnOp([])

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())
