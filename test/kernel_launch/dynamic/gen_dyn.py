from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType

# python gen_relu_1d.py > relu_1d.pto
# ptoas --enable-insert-sync relu_1d.pto > gen_cpp1d.cpp

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            # Specify the tile size and the input matrix height
            tile_h = 2
            tile_w = 128
            H = 20

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view = pto.PartitionTensorViewType.get([tile_h, tile_w], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            # A tile configuration: Simple Row major with no SLayout, 512 fractal size is not used.
            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf = pto.TileBufType.get([tile_h, tile_w], f32, vec, [tile_h, tile_w], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("relu_kernel_1d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                vec_section = pto.SectionVectorOp()
                vec_block = vec_section.body.blocks.append()
                
                with InsertionPoint(vec_block):
                    # constants
                    c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                    c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                    c2 = arith.ConstantOp(IndexType.get(ctx), tile_h).result
                    c20 = arith.ConstantOp(IndexType.get(ctx), H).result
                    c128 = arith.ConstantOp(IndexType.get(ctx), tile_w).result

                    arg0, arg1 = entry.arguments

                    # Get the block_idx and cast it to an index type
                    bid = arith.IndexCastOp(IndexType.get(), pto.GetBlockIdxOp()).result
                    # block_idx i will calculate the relu for rows [c2*i: c2*(i+1)]
                    # basically a tile of shape (c2, 128) at offset c2*i
                    row_offset = arith.MulIOp(bid, c2).result

                    # This is the GlobalTensor is a pointer to the full c20xc128 matrix in GM
                    tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c20, c128], [c128, c1]).result
                    tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c20, c128], [c128, c1]).result

                    # We then point the pointers to the relevant tile region in GM
                    sv0 = pto.PartitionViewOp(tile_view, tv0, offsets=[row_offset, c0], sizes=[c2, c128]).result
                    sv1 = pto.PartitionViewOp(tile_view, tv1, offsets=[row_offset, c0], sizes=[c2, c128]).result

                    # Allocate the tiles in UB memory where the relu calculation will happe
                    tb0 = pto.AllocTileOp(tile_buf).result
                    tb1 = pto.AllocTileOp(tile_buf).result

                    # Load the GM into the tile
                    pto.TLoadOp(None, sv0, tb0)
                    pto.TReluOp(tb0, tb1)
                    # Load the resultant tile back to the output GM
                    pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())