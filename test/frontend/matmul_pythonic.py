# Make matmul_original.py more Pythonic (uses pto_frontend wrappers)

from mlir.ir import (
    Context, Location, InsertionPoint,
    StringAttr,
)
from mlir.dialects import func, scf, builtin
import pto_frontend as pto


def build(
    M=32, K=256, N=32,
    validM=32, validK=256, validN=32,
    BASEK=32,
    # 下面两个 fractal size 你按工程真实 TileConfig 改一下：
    s_fractal_ab=512,
    s_fractal_c=1024,
):
    assert K % BASEK == 0
    iters = K // BASEK

    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        module = builtin.ModuleOp()
        module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")

        # ---- ptr types (Pythonic: pto.tensor(dtype=pto.float32()) ----
        t_f32 = pto.float32()
        ptr_out = pto.tensor(dtype=t_f32)
        ptr_a = pto.tensor(dtype=t_f32)
        ptr_b = pto.tensor(dtype=t_f32)
        ptr_bias = pto.tensor(dtype=t_f32)

        i1 = pto.bool()

        # ---- enum attrs (Pythonic: pto.EVENT_ID0(), etc.) ----
        EVENT_ID0 = pto.EVENT_ID0()
        PIPE_MTE2 = pto.PIPE_MTE2()
        PIPE_MTE1 = pto.PIPE_MTE1()
        PIPE_M = pto.PIPE_M()
        PIPE_FIX = pto.PIPE_FIX()

        # ---- function ----
        # (out, A, B, bias, isBias)
        fn_ty = func.FunctionType.get([ptr_out, ptr_a, ptr_b, ptr_bias, i1], [])
        with InsertionPoint(module.body):
            fn = func.FuncOp("RunTMATMULSplitK", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            out_ptr, a_ptr, b_ptr, bias_ptr, isBias = entry.arguments

            # ---- constants (Pythonic: pto.const(0), etc.) ----
            c0 = pto.const(0)
            c1 = pto.const(1)
            cOne = pto.const(1)

            cM = pto.const(validM)
            cK = pto.const(validK)
            cN = pto.const(validN)

            cBASEK = pto.const(BASEK)
            cIter = pto.const(iters)

            cTileM = pto.const(M)
            cTileN = pto.const(N)

            # ---- make_tensor_view (Pythonic: pto.tensor_view(ptr, ((dims), (strides)), dtype=...)) ----
            tvA = pto.tensor_view(a_ptr, ((cM, cK), (cK, c1)), dtype=t_f32)
            tvB = pto.tensor_view(b_ptr, ((cK, cN), (cN, c1)), dtype=t_f32)
            tvOut = pto.tensor_view(out_ptr, ((cM, cN), (cN, c1)), dtype=t_f32)
            tvBias = pto.tensor_view(bias_ptr, ((cOne, cN), (cN, c1)), dtype=t_f32)

            # ---- alloc tiles (Pythonic: pto.tile(shape, dtype, buf_type=...)) ----
            aMatTile = pto.tile(
                [M, BASEK], t_f32, "mat",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )
            bMatTile = pto.tile(
                [BASEK, N], t_f32, "mat",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )
            biasDataTile = pto.tile(
                [1, N], t_f32, "mat_bias",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )

            aTile = pto.tile(
                [M, BASEK], t_f32, "left",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )
            bTile = pto.tile(
                [BASEK, N], t_f32, "right",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )
            cTile = pto.tile_buf(
                [M, N], t_f32, "acc",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )
            biasTile = pto.tile(
                [1, N], t_f32, "bias",
                s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c,
            )

            # ---- valid dims (passed into ops; alloc has no valid operands) ----
            # 对齐你 C++ TileLeft/Right/Acc/Bias 的 RowValid_/ColValid_

            # ---- loop for split-K ----
            loop = scf.ForOp(c0, cIter, c1, [])
            with InsertionPoint(loop.body):
                i = loop.induction_variable

                # kOff = i * BASEK (TODO: __mul__ overloading for index values)
                kOff = pto.mul(i, cBASEK)

                # subviews for this split-K (Pythonic: pto.subview(tv, tile_shape, offsets, sizes))
                svA = pto.subview(tvA, (M, BASEK), (c0, kOff), (cTileM, cBASEK), dtype=t_f32)
                svB = pto.subview(tvB, (BASEK, N), (kOff, c0), (cBASEK, cTileN), dtype=t_f32)
                svBias = pto.subview(tvBias, (1, N), (c0, c0), (cOne, cTileN), dtype=t_f32)

                # ---- TLOAD (Pythonic: pto.load(sv, tile)) ----
                pto.load(svA, aMatTile)
                pto.load(svB, bMatTile)

                if_load_bias = scf.IfOp(isBias, [], hasElse=True)
                with InsertionPoint(if_load_bias.then_block):
                    pto.load(svBias, biasDataTile)
                    scf.YieldOp([])
                with InsertionPoint(if_load_bias.else_block):
                    scf.YieldOp([])

                # ---- sync: MTE2 -> MTE1 ----
                pto.set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)
                pto.wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)

                # ---- TMOV (Pythonic: pto.move(src, dst)) ----
                pto.move(aMatTile, aTile)
                pto.move(bMatTile, bTile)

                if_mov_bias = scf.IfOp(isBias, [], hasElse=True)
                with InsertionPoint(if_mov_bias.then_block):
                    pto.move(biasDataTile, biasTile)
                    scf.YieldOp([])
                with InsertionPoint(if_mov_bias.else_block):
                    scf.YieldOp([])

                # ---- sync: MTE1 -> M ----
                pto.set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0)
                pto.wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0)

                # ---- i == 0 ? (bias? TMATMUL_BIAS : TMATMUL) : TMATMUL_ACC (TODO: __eq__ overloading) ----
                is_i0 = pto.eq(i, c0)
                if_i0 = scf.IfOp(is_i0, [], hasElse=True)

                # then: i == 0
                with InsertionPoint(if_i0.then_block):
                    if_bias0 = scf.IfOp(isBias, [], hasElse=True)
                    with InsertionPoint(if_bias0.then_block):
                        pto.matmul_bias(aTile, bTile, biasTile, cTile)
                        scf.YieldOp([])
                    with InsertionPoint(if_bias0.else_block):
                        pto.matmul(aTile, bTile, cTile)
                        scf.YieldOp([])
                    scf.YieldOp([])

                # else: i != 0
                with InsertionPoint(if_i0.else_block):
                    pto.matmul_acc(cTile, aTile, bTile, cTile)
                    scf.YieldOp([])

                # ---- sync: M -> MTE2 ----
                pto.set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0)
                pto.wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0)

                scf.YieldOp([])

            # ---- after loop ----
            pto.set_flag(PIPE_M, PIPE_FIX, EVENT_ID0)
            pto.wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0)

            # ---- TSTORE (Pythonic: pto.store(tile, sv)) ----
            svOut = pto.subview(tvOut, (M, N), (c0, c0), (cTileM, cTileN), dtype=t_f32)
            pto.store(cTile, svOut)

            func.ReturnOp([])
        module.operation.verify()
        return module


if __name__ == "__main__":
    m = build()
    print(m)
