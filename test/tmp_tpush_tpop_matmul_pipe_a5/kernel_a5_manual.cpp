#include "pto/pto-inst.hpp"

using namespace pto;

__global__ AICORE void matmul_tpush_tpop_print(__gm__ float *gm_a, __gm__ float *gm_b,
                                                __gm__ float *gm_slot_buffer,
                                                int32_t c2v_consumer_buf,
                                                int32_t v2c_consumer_buf)
{
    (void)gm_slot_buffer;
    (void)v2c_consumer_buf;

    int64_t base0 = 0;
    int64_t base512 = 512;

    Tile<TileType::Mat, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> matA;
    TASSIGN(matA, base0);
    Tile<TileType::Mat, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> matB;
    TASSIGN(matB, base512);

    Tile<TileType::Left, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> leftA;
    TASSIGN(leftA, base0);
    Tile<TileType::Right, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null> rightB;
    TASSIGN(rightB, base0);

    Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null> accC;
    TASSIGN(accC, base0);
    Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null> vecOut;
    TASSIGN(vecOut, base0);

    auto pipe = TPipe<0, FIFOType::VEC_FIFO, 8, 8,
                      Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024,
                           PadValue::Null>,
                      Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512,
                           PadValue::Null>,
                      TSyncOpType::TMOV_C2UB, TSyncOpType::NONE, VecCubeRatio::V2C1_VECS>(c2v_consumer_buf);

#if defined(__DAV_CUBE__)
    using GTShape = pto::Shape<1, 1, 1, 16, 16>;
    using GTStride = pto::Stride<256, 256, 256, 16, 1>;
    using GlobalFloat = GlobalTensor<float, GTShape, GTStride, pto::Layout::ND>;

    GTShape shape = GTShape();
    GTStride stride = GTStride();

    GlobalFloat gA(gm_a, shape, stride);
    GlobalFloat gB(gm_b, shape, stride);

    TLOAD(matA, gA);
    TLOAD(matB, gB);
    TMOV(leftA, matA);
    TMOV(rightB, matB);
    TMATMUL(accC, leftA, rightB);
    TPUSH(accC, pipe);
#endif

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    TPOP(vecOut, pipe);
    TPRINT(vecOut);
    TFREE(pipe);
#endif
}

template <int32_t tilingKey>
void LaunchMatmulTPushPopPrint(uint8_t *a, uint8_t *b, uint8_t *slot,
                               int32_t c2vBuf, int32_t v2cBuf, void *stream)
{
    (void)tilingKey;
    matmul_tpush_tpop_print<<<1, nullptr, stream>>>(reinterpret_cast<float *>(a),
                                                    reinterpret_cast<float *>(b),
                                                    reinterpret_cast<float *>(slot),
                                                    c2vBuf, v2cBuf);
}

template void LaunchMatmulTPushPopPrint<1>(uint8_t *a, uint8_t *b, uint8_t *slot,
                                           int32_t c2vBuf, int32_t v2cBuf, void *stream);
