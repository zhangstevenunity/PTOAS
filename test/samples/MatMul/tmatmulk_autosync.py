from mlir.ir import (
    Context, Location, InsertionPoint,
    IndexType, IntegerType, F16Type, F32Type, StringAttr
)
from mlir.dialects import func, arith, scf, pto, builtin
from mlir.dialects.arith import CmpIPredicate


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build(
    M=32, K=256, N=32,
    validM=32, validK=256, validN=32,
    BASEK=32,
    # 下面两个 fractal size 你按工程真实 TileConfig 改一下：
    s_fractal_ab=512,
    s_fractal_c=1024,
):
    # This sample intentionally contains NO explicit sync ops
    # (record_event/wait_event or set_flag/wait_flag).
    #
    # It is meant to be used with:
    #   ptoas --enable-insert-sync
    #
    # to exercise the auto-sync insertion + event-id allocation pipeline.
    assert K % BASEK == 0
    iters = K // BASEK

    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        module = builtin.ModuleOp()
        module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")

        # ---- element types ----
        t_out = F32Type.get()
        t_a = F32Type.get()
        t_b = F32Type.get()
        t_bias = F32Type.get()

        # ---- ptr types ----
        ptr_out = pto.PtrType.get(t_out)
        ptr_a = pto.PtrType.get(t_a)
        ptr_b = pto.PtrType.get(t_b)
        ptr_bias = pto.PtrType.get(t_bias)

        i1 = IntegerType.get_signless(1)

        # ---- tensor view types ----
        tv2_a = pto.TensorViewType.get(2, t_a)        # [validM, validK]
        tv2_b = pto.TensorViewType.get(2, t_b)        # [validK, validN]
        tv2_out = pto.TensorViewType.get(2, t_out)    # [validM, validN]
        tv2_bias = pto.TensorViewType.get(2, t_bias)  # [1, validN]

        # ---- tile view types ----
        tile_view_a = pto.PartitionTensorViewType.get([M, BASEK], t_a)
        tile_view_b = pto.PartitionTensorViewType.get([BASEK, N], t_b)
        tile_view_out = pto.PartitionTensorViewType.get([M, N], t_out)
        tile_view_bias = pto.PartitionTensorViewType.get([1, N], t_bias)

        # ---- address spaces ----
        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT)
        left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT)
        right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT)
        acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC)
        bias = pto.AddressSpaceAttr.get(pto.AddressSpace.BIAS)

        # ---- configs (3rd arg = s_fractal_size) ----
        cfg_mat = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        cfg_mat_bias = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.NoneBox),
            s_fractal_ab,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        cfg_left = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        cfg_right = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            s_fractal_ab,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        cfg_acc = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_c,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        cfg_bias = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.NoneBox),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        # ---- tile buf types (each has its own cfg) ----
        tile_buf_aMat = pto.TileBufType.get([M, BASEK], t_a, mat, [M, BASEK], cfg_mat)
        tile_buf_bMat = pto.TileBufType.get([BASEK, N], t_b, mat, [BASEK, N], cfg_mat)
        tile_buf_biasData = pto.TileBufType.get([1, N], t_bias, mat, [1, N], cfg_mat_bias)

        tile_buf_aTile = pto.TileBufType.get([M, BASEK], t_a, left, [M, BASEK], cfg_left)
        tile_buf_bTile = pto.TileBufType.get([BASEK, N], t_b, right, [BASEK, N], cfg_right)
        tile_buf_cTile = pto.TileBufType.get([M, N], t_out, acc, [M, N], cfg_acc)
        tile_buf_biasTile = pto.TileBufType.get([1, N], t_bias, bias, [1, N], cfg_bias)

        # ---- function ----
        # (out, A, B, bias, isBias)
        fn_ty = func.FunctionType.get([ptr_out, ptr_a, ptr_b, ptr_bias, i1], [])
        with InsertionPoint(module.body):
            fn = func.FuncOp("RunTMATMULSplitK", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            out_ptr, a_ptr, b_ptr, bias_ptr, isBias = entry.arguments

            # ---- constants ----
            c0 = _idx_const(0)
            c1 = _idx_const(1)
            cOne = _idx_const(1)

            cM = _idx_const(validM)
            cK = _idx_const(validK)
            cN = _idx_const(validN)

            cBASEK = _idx_const(BASEK)
            cIter = _idx_const(iters)

            cTileM = _idx_const(M)
            cTileN = _idx_const(N)

            # ---- make_tensor_view ----
            tvA = pto.MakeTensorViewOp(tv2_a, a_ptr, [cM, cK], [cK, c1]).result
            tvB = pto.MakeTensorViewOp(tv2_b, b_ptr, [cK, cN], [cN, c1]).result
            tvOut = pto.MakeTensorViewOp(tv2_out, out_ptr, [cM, cN], [cN, c1]).result
            tvBias = pto.MakeTensorViewOp(tv2_bias, bias_ptr, [cOne, cN], [cN, c1]).result

            # ---- alloc tiles ----
            aMatTile = pto.AllocTileOp(tile_buf_aMat).result
            bMatTile = pto.AllocTileOp(tile_buf_bMat).result
            biasDataTile = pto.AllocTileOp(tile_buf_biasData).result

            aTile = pto.AllocTileOp(tile_buf_aTile).result
            bTile = pto.AllocTileOp(tile_buf_bTile).result
            cTile = pto.AllocTileOp(tile_buf_cTile).result
            biasTile = pto.AllocTileOp(tile_buf_biasTile).result

            # ---- loop for split-K ----
            loop = scf.ForOp(c0, cIter, c1, [])
            with InsertionPoint(loop.body):
                i = loop.induction_variable

                kOff = arith.MulIOp(i, cBASEK).result

                svA = pto.PartitionViewOp(
                    tile_view_a, tvA, offsets=[c0, kOff], sizes=[cTileM, cBASEK]
                ).result
                svB = pto.PartitionViewOp(
                    tile_view_b, tvB, offsets=[kOff, c0], sizes=[cBASEK, cTileN]
                ).result
                svBias = pto.PartitionViewOp(
                    tile_view_bias, tvBias, offsets=[c0, c0], sizes=[cOne, cTileN]
                ).result

                # ---- TLOAD ----
                pto.TLoadOp(None, svA, aMatTile)
                pto.TLoadOp(None, svB, bMatTile)

                if_load_bias = scf.IfOp(isBias, [], hasElse=True)
                with InsertionPoint(if_load_bias.then_block):
                    pto.TLoadOp(None, svBias, biasDataTile)
                    scf.YieldOp([])
                with InsertionPoint(if_load_bias.else_block):
                    scf.YieldOp([])

                # ---- TMOV ----
                pto.TMovOp(None, aMatTile, aTile)
                pto.TMovOp(None, bMatTile, bTile)

                if_mov_bias = scf.IfOp(isBias, [], hasElse=True)
                with InsertionPoint(if_mov_bias.then_block):
                    pto.TMovOp(None, biasDataTile, biasTile)
                    scf.YieldOp([])
                with InsertionPoint(if_mov_bias.else_block):
                    scf.YieldOp([])

                # ---- i == 0 ? (bias? TMATMUL_BIAS : TMATMUL) : TMATMUL_ACC ----
                is_i0 = arith.CmpIOp(CmpIPredicate.eq, i, c0).result
                if_i0 = scf.IfOp(is_i0, [], hasElse=True)

                with InsertionPoint(if_i0.then_block):
                    if_bias0 = scf.IfOp(isBias, [], hasElse=True)
                    with InsertionPoint(if_bias0.then_block):
                        pto.TMatmulBiasOp(None, aTile, bTile, biasTile, cTile)
                        scf.YieldOp([])
                    with InsertionPoint(if_bias0.else_block):
                        pto.TMatmulOp(None, aTile, bTile, cTile)
                        scf.YieldOp([])
                    scf.YieldOp([])

                with InsertionPoint(if_i0.else_block):
                    pto.TMatmulAccOp(None, cTile, aTile, bTile, cTile)
                    scf.YieldOp([])

                scf.YieldOp([])

            # ---- after loop ----
            svOut = pto.PartitionViewOp(
                tile_view_out, tvOut, offsets=[c0, c0], sizes=[cTileM, cTileN]
            ).result
            pto.TStoreOp(None, cTile, svOut)

            func.ReturnOp([])

        module.operation.verify()
        return module


if __name__ == "__main__":
    m = build()
    print(m)
