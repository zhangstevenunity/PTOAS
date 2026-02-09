"""
TMrgSortOp format2: ins(src0..src3) outs(dst, excuted) with exhausted attr.
  - 4 tile_buf srcs, 2 dsts (tile_buf + vector<4xi16> excuted), optional exhausted bool.
"""
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType, IntegerType

from mlir.ir import VectorType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            i16 = IntegerType.get_signless(16, ctx)
            # Format2 excuted: vector<4xi16>
            vec_4_i16 = VectorType.get([4], i16)

            tv2_f32 = pto.TensorViewType.get([32, 32], f32, ctx)
            part_view_32x32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_1x1024 = pto.TileBufType.get([1, 1024], f32, vec, [1, 1024], cfg, ctx)

            # Kernel: (in_ptr, out_ptr, excuted_vector) -> ()
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, vec_4_i16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("mrgsort_format2_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                c256 = arith.ConstantOp(IndexType.get(ctx), 256).result

                arg0, arg1, excuted = entry.arguments

                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c256, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c256, c1]).result
                sv0 = pto.PartitionViewOp(part_view_32x32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result

                # 4 src tiles + 1 dst tile + 1 excuted (vector<4xi16>)
                tb_s0 = pto.AllocTileOp(tile_buf_1x1024).result
                tb_s1 = pto.AllocTileOp(tile_buf_1x1024).result
                tb_s2 = pto.AllocTileOp(tile_buf_1x1024).result
                tb_s3 = pto.AllocTileOp(tile_buf_1x1024).result
                tb_dst = pto.AllocTileOp(tile_buf_1x1024).result

                pto.TLoadOp(None, sv0, tb_s0)
                # Format2: ins(src0..src3) outs(dst) excuted=vector, exhausted=false
                pto.TMrgSortOp(
                    srcs=[tb_s0, tb_s1, tb_s2, tb_s3],
                    dsts=[tb_dst],
                    excuted=excuted,
                    exhausted=False,
                )

                sv1 = pto.PartitionViewOp(part_view_32x32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result
                pto.TStoreOp(None, tb_dst, sv1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
