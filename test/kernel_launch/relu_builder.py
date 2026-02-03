# Adapted from https://github.com/zhangstevenunity/PTOAS/blob/main/test/samples/Sync/sync.py

from mlir.ir import Context, Location, Module, InsertionPoint, Attribute
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        m = Module.create()

        f32 = F32Type.get()
        ptr_f32 = pto.PtrType.get(f32)

        tv2_f32 = pto.TensorViewType.get(2, f32)
        tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32)
        ub = pto.AddressSpaceAttr.get(pto.AddressSpace.UB)
        bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor)
        sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox)
        pd = pto.PadValueAttr.get(pto.PadValue.Null)

        cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd)
        tile_buf_32 = pto.TileBufType.get([32, 32], f32, ub, [32, 32], cfg)

        PIPE_MTE2 = Attribute.parse("#pto.pipe<PIPE_MTE2>")
        PIPE_V = Attribute.parse("#pto.pipe<PIPE_V>")
        PIPE_MTE3 = Attribute.parse("#pto.pipe<PIPE_MTE3>")
        EVENT_ID0 = Attribute.parse("#pto.event<EVENT_ID0>")

        fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
        with InsertionPoint(m.body):
            fn = func.FuncOp("sync_kernel_2d", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            c0 = arith.ConstantOp(IndexType.get(), 0).result
            c1 = arith.ConstantOp(IndexType.get(), 1).result
            c32 = arith.ConstantOp(IndexType.get(), 32).result

            arg0, arg1 = entry.arguments

            tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
            tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result

            sv0 = pto.PartitionViewOp(tile_view_32, tv0, [c0, c0], [c32, c32]).result
            sv1 = pto.PartitionViewOp(tile_view_32, tv1, [c0, c0], [c32, c32]).result

            tb0 = pto.AllocTileOp(tile_buf_32).result
            tb1 = pto.AllocTileOp(tile_buf_32).result

            pto.TLoadOp(None, sv0, tb0)
            pto.SetFlagOp(PIPE_MTE2, PIPE_V, EVENT_ID0)
            pto.WaitFlagOp(PIPE_MTE2, PIPE_V, EVENT_ID0)

            pto.TReluOp(tb0, tb1)
            pto.SetFlagOp(PIPE_V, PIPE_MTE3, EVENT_ID0)
            pto.WaitFlagOp(PIPE_V, PIPE_MTE3, EVENT_ID0)

            pto.TStoreOp(None, tb1, sv1)
            func.ReturnOp([])

        m.operation.verify()
        return m


if __name__ == "__main__":
    print(build())
