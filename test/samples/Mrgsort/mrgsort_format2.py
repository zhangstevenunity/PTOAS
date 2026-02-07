"""
TMrgSortOp format2: ins(src0..src3) outs(dst, tmp, excuted) with exhausted attr.

Important notes for on-device execution:
  - pto-isa's TMRGSORT expects 4 input lists; leaving src1..src3 uninitialized
    can lead to illegal configurations and runtime exceptions on NPU.
  - This testcase therefore loads 4 independent GM inputs into src tiles.
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

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            # Each list holds 128 packed structures (value+index), i.e. 256 f32 words.
            # Represent it as 8x32 to keep strides simple.
            part_view_8x32 = pto.PartitionTensorViewType.get([8, 32], f32, ctx)
            part_view_32x32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_1x256 = pto.TileBufType.get([1, 256], f32, vec, [1, 256], cfg, ctx)
            tile_buf_1x1024 = pto.TileBufType.get([1, 1024], f32, vec, [1, 1024], cfg, ctx)

            # Kernel: (in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr, executed_list) -> ()
            fn_ty = func.FunctionType.get(
                [ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f32, vec_4_i16], []
            )
            with InsertionPoint(m.body):
                fn = func.FuncOp("mrgsort_format2_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c8 = arith.ConstantOp(IndexType.get(ctx), 8).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                c256 = arith.ConstantOp(IndexType.get(ctx), 256).result

                arg0, arg1, arg2, arg3, arg_out, excuted = entry.arguments

                # Inputs: 8x32 (256 f32) per list.
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c8, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c8, c32], [c32, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c8, c32], [c32, c1]).result
                tv3 = pto.MakeTensorViewOp(tv2_f32, arg3, [c8, c32], [c32, c1]).result
                # Output: 32x32 (1024 f32) for 512 packed structures.
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c32, c32], [c256, c1]).result

                sv0 = pto.PartitionViewOp(part_view_8x32, tv0, offsets=[c0, c0], sizes=[c8, c32]).result
                sv1 = pto.PartitionViewOp(part_view_8x32, tv1, offsets=[c0, c0], sizes=[c8, c32]).result
                sv2 = pto.PartitionViewOp(part_view_8x32, tv2, offsets=[c0, c0], sizes=[c8, c32]).result
                sv3 = pto.PartitionViewOp(part_view_8x32, tv3, offsets=[c0, c0], sizes=[c8, c32]).result

                # Format2: 4 src tiles + 2 dst tiles (dst + tmp) + 1 executed list.
                tb_s0 = pto.AllocTileOp(tile_buf_1x256).result
                tb_s1 = pto.AllocTileOp(tile_buf_1x256).result
                tb_s2 = pto.AllocTileOp(tile_buf_1x256).result
                tb_s3 = pto.AllocTileOp(tile_buf_1x256).result
                tb_dst = pto.AllocTileOp(tile_buf_1x1024).result
                tb_tmp = pto.AllocTileOp(tile_buf_1x1024).result

                pto.TLoadOp(None, sv0, tb_s0)
                pto.TLoadOp(None, sv1, tb_s1)
                pto.TLoadOp(None, sv2, tb_s2)
                pto.TLoadOp(None, sv3, tb_s3)

                # Format2: ins(src0..src3) outs(dst, tmp) excuted=vector, exhausted=false
                pto.TMrgSortOp(
                    srcs=[tb_s0, tb_s1, tb_s2, tb_s3],
                    dsts=[tb_dst, tb_tmp],
                    excuted=excuted,
                    exhausted=False,
                )

                sv_out = pto.PartitionViewOp(part_view_32x32, tv_out, offsets=[c0, c0], sizes=[c32, c32]).result
                pto.TStoreOp(None, tb_dst, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
