"""
Build PTO dialect IR for the AICORE `runTEXTRACT` flow:
TLOAD(src0/src1) -> TEXTRACT(a/b sub-tiles) -> TMATMUL -> TSTORE(out)

Reference:
  - pto-isa/tests/npu/a5/src/st/testcase/textract/textract_kernel.cpp
  - pto-as/test/samples/MatMul/tmatmulk.py
"""

from mlir.ir import Context, Location, InsertionPoint, IndexType, IntegerType, F32Type, StringAttr
from mlir.dialects import func, arith, scf, pto, builtin
from mlir.dialects.pto import TLOAD, TMOV_M2L, TMATMUL, TSTORE_ACC, EVENT_ID0


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build(
    M=32,
    K=256,
    N=32,
    indexM=0,
    indexK=0,
    indexN=0,
    s_fractal_ab=512,
    s_fractal_c=1024,
):
    """
    RunTEXTRACT(out, src0, src1, isAtranspose, isBtranspose):
    isAtranspose/isBtranspose 为运行时参数，用于选择 src0/src1 的布局与 MAT tile 配置。
    """
    if not (0 <= indexM <= M and 0 <= indexK <= K and 0 <= indexN <= N):
        raise ValueError("indexM/indexK/indexN must be within the input tensor bounds")

    m_valid = M - indexM
    k_valid = K - indexK
    n_valid = N - indexN

    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        module = builtin.ModuleOp()
        module.attributes["pto.target_arch"] = StringAttr.get("a3")

        t_out = F32Type.get()
        t_a = F32Type.get()
        t_b = F32Type.get()
        i1 = IntegerType.get_signless(1)

        ptr_out = pto.PtrType.get(t_out)
        ptr_src0 = pto.PtrType.get(t_a)
        ptr_src1 = pto.PtrType.get(t_b)

        tv2_a = pto.TensorViewType.get(2, t_a)
        tv2_b = pto.TensorViewType.get(2, t_b)
        tv2_out = pto.TensorViewType.get(2, t_out)

        tile_view_a = pto.PartitionTensorViewType.get([M, K], t_a)
        tile_view_b = pto.PartitionTensorViewType.get([K, N], t_b)
        tile_view_out = pto.PartitionTensorViewType.get([m_valid, n_valid], t_out)

        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT)
        left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT)
        right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT)
        acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC)

        pad_null = pto.PadValueAttr.get(pto.PadValue.Null)

        # A Mat: ND = ColMajor/RowMajor, DN = RowMajor/ColMajor (与 C++ isAtranspose 一致)
        cfg_a_mat_nd = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,
            pad_null,
        )
        cfg_a_mat_dn = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            s_fractal_ab,
            pad_null,
        )
        # B Mat: ND = ColMajor/RowMajor, DN = RowMajor/ColMajor (与 C++ isBtranspose 一致)
        cfg_b_mat_nd = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,
            pad_null,
        )
        cfg_b_mat_dn = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            s_fractal_ab,
            pad_null,
        )
        cfg_left = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,
            pad_null,
        )
        cfg_right = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            s_fractal_ab,
            pad_null,
        )
        cfg_acc = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_c,
            pad_null,
        )

        tile_buf_a_mat_nd = pto.TileBufType.get([M, K], t_a, mat, [M, K], cfg_a_mat_nd)
        tile_buf_a_mat_dn = pto.TileBufType.get([M, K], t_a, mat, [M, K], cfg_a_mat_dn)
        tile_buf_b_mat_nd = pto.TileBufType.get([K, N], t_b, mat, [K, N], cfg_b_mat_nd)
        tile_buf_b_mat_dn = pto.TileBufType.get([K, N], t_b, mat, [K, N], cfg_b_mat_dn)
        tile_buf_a_tile = pto.TileBufType.get(
            [m_valid, k_valid], t_a, left, [m_valid, k_valid], cfg_left
        )
        tile_buf_b_tile = pto.TileBufType.get(
            [k_valid, n_valid], t_b, right, [k_valid, n_valid], cfg_right
        )
        tile_buf_c_tile = pto.TileBufType.get(
            [m_valid, n_valid], t_out, acc, [m_valid, n_valid], cfg_acc
        )

        # 函数签名: (out, src0, src1, isAtranspose, isBtranspose)
        fn_ty = func.FunctionType.get(
            [ptr_out, ptr_src0, ptr_src1, i1, i1], []
        )
        with InsertionPoint(module.body):
            fn = func.FuncOp("RunTEXTRACT", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            out_ptr, src0_ptr, src1_ptr, is_a_transpose, is_b_transpose = entry.arguments

            c0 = _idx_const(0)
            c1 = _idx_const(1)
            cM = _idx_const(M)
            cK = _idx_const(K)
            cN = _idx_const(N)
            cMValid = _idx_const(m_valid)
            cKValid = _idx_const(k_valid)
            cNValid = _idx_const(n_valid)
            cIndexM = _idx_const(indexM)
            cIndexK = _idx_const(indexK)
            cIndexN = _idx_const(indexN)

            layout_nd = pto.LayoutAttr.get(pto.Layout.ND)
            layout_dn = pto.LayoutAttr.get(pto.Layout.DN)

            # src0 [M,K]: ND = row-major stride [K,1], DN = col-major stride [1,M] (与 C++ GlobalDataSrc0 一致)
            tv_src0_nd = pto.MakeTensorViewOp(
                tv2_a, src0_ptr, [cM, cK], [cK, c1], layout=layout_nd
            ).result
            tv_src0_dn = pto.MakeTensorViewOp(
                tv2_a, src0_ptr, [cM, cK], [c1, cM], layout=layout_dn
            ).result
            # src1 [K,N]: DN = col-major stride [1,K], ND = row-major stride [N,1] (与 C++ GlobalDataSrc1 一致)
            tv_src1_nd = pto.MakeTensorViewOp(
                tv2_b, src1_ptr, [cK, cN], [cN, c1], layout=layout_nd
            ).result
            tv_src1_dn = pto.MakeTensorViewOp(
                tv2_b, src1_ptr, [cK, cN], [c1, cK], layout=layout_dn
            ).result
            tv_out = pto.MakeTensorViewOp(
                tv2_out, out_ptr, [cMValid, cNValid], [cNValid, c1]
            ).result

            sv_a_nd = pto.PartitionViewOp(
                tile_view_a, tv_src0_nd, offsets=[c0, c0], sizes=[cM, cK]
            ).result
            sv_a_dn = pto.PartitionViewOp(
                tile_view_a, tv_src0_dn, offsets=[c0, c0], sizes=[cM, cK]
            ).result
            sv_b_nd = pto.PartitionViewOp(
                tile_view_b, tv_src1_nd, offsets=[c0, c0], sizes=[cK, cN]
            ).result
            sv_b_dn = pto.PartitionViewOp(
                tile_view_b, tv_src1_dn, offsets=[c0, c0], sizes=[cK, cN]
            ).result

            a_mat_nd = pto.AllocTileOp(tile_buf_a_mat_nd).result
            a_mat_dn = pto.AllocTileOp(tile_buf_a_mat_dn).result
            b_mat_nd = pto.AllocTileOp(tile_buf_b_mat_nd).result
            b_mat_dn = pto.AllocTileOp(tile_buf_b_mat_dn).result
            a_tile = pto.AllocTileOp(tile_buf_a_tile).result
            b_tile = pto.AllocTileOp(tile_buf_b_tile).result
            c_tile = pto.AllocTileOp(tile_buf_c_tile).result

            # ---- TLOAD: 按 isAtranspose / isBtranspose 选 view 与 MAT tile ----
            if_a = scf.IfOp(is_a_transpose, [], hasElse=True)
            with InsertionPoint(if_a.then_block):
                pto.TLoadOp(None, sv_a_dn, a_mat_dn)
                scf.YieldOp([])
            with InsertionPoint(if_a.else_block):
                pto.TLoadOp(None, sv_a_nd, a_mat_nd)
                scf.YieldOp([])

            if_b = scf.IfOp(is_b_transpose, [], hasElse=True)
            with InsertionPoint(if_b.then_block):
                pto.TLoadOp(None, sv_b_nd, b_mat_nd)
                scf.YieldOp([])
            with InsertionPoint(if_b.else_block):
                pto.TLoadOp(None, sv_b_dn, b_mat_dn)
                scf.YieldOp([])

            pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
            pto.wait_event(TLOAD, TMOV_M2L, EVENT_ID0)

            # ---- TEXTRACT: 按 isAtranspose / isBtranspose 从对应 MAT 取到 L0 ----
            if_a_ext = scf.IfOp(is_a_transpose, [], hasElse=True)
            with InsertionPoint(if_a_ext.then_block):
                pto.TExtractOp(a_mat_dn, cIndexM, cIndexK, a_tile)
                scf.YieldOp([])
            with InsertionPoint(if_a_ext.else_block):
                pto.TExtractOp(a_mat_nd, cIndexM, cIndexK, a_tile)
                scf.YieldOp([])

            if_b_ext = scf.IfOp(is_b_transpose, [], hasElse=True)
            with InsertionPoint(if_b_ext.then_block):
                pto.TExtractOp(b_mat_nd, cIndexK, cIndexN, b_tile)
                scf.YieldOp([])
            with InsertionPoint(if_b_ext.else_block):
                pto.TExtractOp(b_mat_dn, cIndexK, cIndexN, b_tile)
                scf.YieldOp([])

            pto.record_event(TMOV_M2L, TMATMUL, EVENT_ID0)
            pto.wait_event(TMOV_M2L, TMATMUL, EVENT_ID0)

            pto.TMatmulOp(None, a_tile, b_tile, c_tile)

            pto.record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
            pto.wait_event(TMATMUL, TSTORE_ACC, EVENT_ID0)

            sv_out = pto.PartitionViewOp(
                tile_view_out, tv_out, offsets=[c0, c0], sizes=[cMValid, cNValid]
            ).result
            pto.TStoreOp(None, c_tile, sv_out)

            func.ReturnOp([])

        module.operation.verify()
        return module


if __name__ == "__main__":
    # 生成 RunTEXTRACT(out, src0, src1, isAtranspose, isBtranspose) 的 IR
    print(build())
