from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto, scf
from mlir.ir import F32Type, IndexType, IntegerType

# python gen_dyn.py | ptoas --enable-insert-sync > generated_relu.cpp

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            tile_w = 32
            NUM_BLOCKS = 20

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
                vec_section = pto.SectionVectorOp()
                vec_block = vec_section.body.blocks.append()
                
                with InsertionPoint(vec_block):
                    arg0, arg1, argN = entry.arguments
                    c0 = arith.ConstantOp(idx, 0).result
                    c1 = arith.ConstantOp(idx, 1).result
                    c_tile_w = arith.ConstantOp(idx, tile_w).result
                    c_num_blocks = arith.ConstantOp(u32, NUM_BLOCKS).result
                    total_elements = arith.IndexCastOp(idx, argN).result

                    # must do divisions in u32 for now. until index is supported
                    num_el_per_core = arith.DivUIOp(argN, c_num_blocks).result
                    num_tiles = arith.DivUIOp(
                        num_el_per_core, arith.IndexCastOp(u32, c_tile_w).result
                    ).result
                    num_el_per_core = arith.IndexCastOp(idx, num_el_per_core).result
                    num_tiles = arith.IndexCastOp(idx, num_tiles).result
                    bid = arith.IndexCastOp(idx, pto.GetBlockIdxOp()).result

                    # GM tensors shape N with stride 1
                    tv0 = pto.MakeTensorViewOp(tv1_f32, arg0, [total_elements], [c1]).result
                    tv1 = pto.MakeTensorViewOp(tv1_f32, arg1, [total_elements], [c1]).result

                    # allocate tiles in UB
                    tb0 = pto.AllocTileOp(tile_buf).result
                    tb1 = pto.AllocTileOp(tile_buf).result

                    # for loop: for i in range(num_tiles)
                    loop = scf.ForOp(c0, num_tiles, c1)
                    with InsertionPoint(loop.body):
                        i = loop.induction_variable
                        offset_core = arith.MulIOp(bid, num_el_per_core).result
                        offset_tile = arith.MulIOp(i, c_tile_w).result
                        offset_total = arith.AddIOp(offset_core, offset_tile).result

                        # each core c takes a tile at offset c*nun_el_per_core+i*tile_w  
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