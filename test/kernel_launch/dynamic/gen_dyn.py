from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto, scf
from mlir.ir import F32Type, IndexType, IntegerType

# python gen_relu_1d.py > relu_1d.pto
# ptoas --enable-insert-sync relu_1d.pto > gen_cpp1d.cpp

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            tile_w = 32
            NUM_BLOCKS = 10

            f32 = F32Type.get(ctx)
            u32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv1_f32 = pto.TensorViewType.get(1, f32, ctx)
            tile_view = pto.PartitionTensorViewType.get([tile_w], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf = pto.TileBufType.get([1, tile_w], f32, vec, [1, tile_w], cfg, ctx)

            # function signature: (float*, float*, uint32 N)
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, u32], [])

            with InsertionPoint(m.body):
                fn = func.FuncOp("sync_kernel_dyn", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                arg0, arg1, argN = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c_tile_w = arith.ConstantOp(idx, tile_w).result
                c_num_blocks = arith.ConstantOp(idx, NUM_BLOCKS).result

                # cast N to index
                N_idx = arith.IndexCastOp(idx, argN).result
                # num_el_per_core = N / NUM_BLOCKS
                num_el_per_core = arith.DivUIOp(N_idx, c_num_blocks).result
                # num_tiles = num_el_per_core / tile_w
                num_tiles = arith.DivUIOp(num_el_per_core, c_tile_w).result
                bid = arith.IndexCastOp(idx, pto.GetBlockIdxOp()).result

                # GM tensors shape N with stride 1
                tv0 = pto.MakeTensorViewOp(tv1_f32, arg0, [N_idx], [c1]).result
                tv1 = pto.MakeTensorViewOp(tv1_f32, arg1, [N_idx], [c1]).result

                # allocate tiles in UB
                tb0 = pto.AllocTileOp(tile_buf).result
                tb1 = pto.AllocTileOp(tile_buf).result

                # for loop: for i in range(num_tiles)
                loop = scf.ForOp(c0, num_tiles, c1)
                with InsertionPoint(loop.body):
                    i = loop.induction_variable
                    """
                    1000 elements. 100 el per core:

                    core 2 has [200, 300]

                    then for tile offsets we want tile_w*i
                    """
                    offset_core = arith.MulIOp(bid, num_el_per_core).result
                    offset_tile = arith.MulIOp(i, c_tile_w).result
                    offset_total = arith.AddIOp(offset_core, offset_tile).result

                    sv0 = pto.PartitionViewOp(
                        tile_view,
                        tv0,
                        offsets=[offset_total],
                        sizes=[c_tile_w]
                    ).result
                    sv1 = pto.PartitionViewOp(
                        tile_view,
                        tv1,
                        offsets=[offset_total],
                        sizes=[c_tile_w]
                    ).result

                    pto.TLoadOp(None, sv0, tb0)
                    pto.TReluOp(tb0, tb1)
                    pto.TStoreOp(None, tb1, sv1)

                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m

if __name__ == "__main__":
    print(build())