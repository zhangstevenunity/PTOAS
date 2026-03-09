#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <int32_t tilingKey>
void LaunchMatmulTPushPopPrint(uint8_t *a, uint8_t *b, uint8_t *slot,
                               int32_t c2vBuf, int32_t v2cBuf, void *stream);

#define ACL_CHECK(expr)                                                                                           \
    do {                                                                                                          \
        aclError _ret = (expr);                                                                                   \
        if (_ret != ACL_SUCCESS) {                                                                                \
            std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            return 1;                                                                                             \
        }                                                                                                         \
    } while (0)

int main()
{
    constexpr int M = 16;
    constexpr int K = 16;
    constexpr int N = 16;
    constexpr size_t aBytes = M * K * sizeof(float);
    constexpr size_t bBytes = K * N * sizeof(float);
    constexpr size_t slotBytes = M * N * sizeof(float);

    std::vector<float> hostA(M * K, 0.0f);
    std::vector<float> hostB(K * N, 1.0f);
    std::vector<float> hostSlot(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        hostA[i * K + i] = 1.0f; // A = I
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    uint8_t *devA = nullptr;
    uint8_t *devB = nullptr;
    uint8_t *devSlot = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devA), aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devB), bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devSlot), slotBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(devA, aBytes, hostA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devB, bBytes, hostB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devSlot, slotBytes, hostSlot.data(), slotBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // For A5 C2V VEC_FIFO path, c2vBuf is local UB base address (example value).
    constexpr int32_t c2vBuf = 0x10000;
    constexpr int32_t v2cBuf = 0;

    LaunchMatmulTPushPopPrint<1>(devA, devB, devSlot, c2vBuf, v2cBuf, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::puts("Kernel finished. Expect TPRINT output to be all 1.0 (A=I, B=all-ones).");

    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devSlot);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
