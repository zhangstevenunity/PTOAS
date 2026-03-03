from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            # NOTE: Regression test for Issue #170.
            #
            # PTO-ISA `TRESHAPE` aliases dst to src (pointer reassignment) on NPU.
            # If PTOAS models `treshape` as writing into a separately-allocated
            # dst tile buffer, MemPlan may over-allocate UB and/or reuse src
            # memory incorrectly. This sample intentionally creates many reshape
            # "views" that must not require additional UB allocations.

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("reshape_alias_memreuse_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1 = entry.arguments

                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result

                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result

                # Base tile: load from GM.
                tb0 = pto.AllocTileOp(tile_buf_32).result
                pto.TLoadOp(None, sv0, tb0)

                # Create many reshape "views" and delay their uses until after
                # all reshapes, so that (without correct alias modeling) UB
                # allocation would exceed the device limit.
                outs = []
                for _ in range(48):
                    tb = pto.AllocTileOp(tile_buf_32).result
                    pto.TReshapeOp(tb0, tb)
                    outs.append(tb)

                # Use all outputs at the end to keep their live ranges overlapping.
                for tb in outs:
                    pto.TStoreOp(None, tb, sv1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

