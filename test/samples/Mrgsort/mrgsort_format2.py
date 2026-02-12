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
            # Each list holds packed (value, index) structures. In pto-isa each
            # structure is 8 bytes (2 x f32 when using float tiles). Keep the
            # testcase small enough to satisfy device constraints across SoCs.
            #
            # 4 lists * 64 structures/list = 256 structures output.
            # 1 structure = 2 floats => 128 floats/list, 512 floats output.
            part_view_1x128 = pto.PartitionTensorViewType.get([1, 128], f32, ctx)
            part_view_1x512 = pto.PartitionTensorViewType.get([1, 512], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_1x128 = pto.TileBufType.get([1, 128], f32, vec, [1, 128], cfg, ctx)
            tile_buf_1x512 = pto.TileBufType.get([1, 512], f32, vec, [1, 512], cfg, ctx)

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
                c128 = arith.ConstantOp(IndexType.get(ctx), 128).result
                c512 = arith.ConstantOp(IndexType.get(ctx), 512).result

                arg0, arg1, arg2, arg3, arg_out, excuted = entry.arguments

                # Inputs: 1x128 (128 f32) per list.
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c1, c128], [c128, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c1, c128], [c128, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c1, c128], [c128, c1]).result
                tv3 = pto.MakeTensorViewOp(tv2_f32, arg3, [c1, c128], [c128, c1]).result
                # Output: 1x512 (512 f32) for 256 packed structures.
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c1, c512], [c512, c1]).result

                sv0 = pto.PartitionViewOp(part_view_1x128, tv0, offsets=[c0, c0], sizes=[c1, c128]).result
                sv1 = pto.PartitionViewOp(part_view_1x128, tv1, offsets=[c0, c0], sizes=[c1, c128]).result
                sv2 = pto.PartitionViewOp(part_view_1x128, tv2, offsets=[c0, c0], sizes=[c1, c128]).result
                sv3 = pto.PartitionViewOp(part_view_1x128, tv3, offsets=[c0, c0], sizes=[c1, c128]).result

                # Format2: 4 src tiles + 2 dst tiles (dst + tmp) + 1 executed list.
                tb_s0 = pto.AllocTileOp(tile_buf_1x128).result
                tb_s1 = pto.AllocTileOp(tile_buf_1x128).result
                tb_s2 = pto.AllocTileOp(tile_buf_1x128).result
                tb_s3 = pto.AllocTileOp(tile_buf_1x128).result
                tb_dst = pto.AllocTileOp(tile_buf_1x512).result
                tb_tmp = pto.AllocTileOp(tile_buf_1x512).result

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

                sv_out = pto.PartitionViewOp(part_view_1x512, tv_out, offsets=[c0, c0], sizes=[c1, c512]).result
                pto.TStoreOp(None, tb_dst, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
