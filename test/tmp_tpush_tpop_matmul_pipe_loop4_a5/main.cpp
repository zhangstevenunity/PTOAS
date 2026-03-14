#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void LaunchMatmulTPushPopLoop4Print(uint8_t *a, uint8_t *b_all, uint8_t *slot,
                                    int32_t c2vBuf, void *stream);

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclError _ret = (expr);                                                    \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  constexpr int M = 16;
  constexpr int K = 16;
  constexpr int N = 16;
  constexpr int Iter = 4;
  constexpr size_t aBytes = M * K * sizeof(float);
  constexpr size_t bBytes = Iter * K * N * sizeof(float);
  constexpr size_t slotBytes = M * N * sizeof(float);

  std::vector<float> hostA(M * K, 0.0f);
  std::vector<float> hostBAll(Iter * K * N, 0.0f);
  std::vector<float> hostSlot(M * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    hostA[i * K + i] = 1.0f;
  }
  for (int iter = 0; iter < Iter; ++iter) {
    const float value = static_cast<float>(iter + 1);
    const size_t base = static_cast<size_t>(iter) * K * N;
    for (int idx = 0; idx < K * N; ++idx) {
      hostBAll[base + idx] = value;
    }
  }

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  uint8_t *devA = nullptr;
  uint8_t *devBAll = nullptr;
  uint8_t *devSlot = nullptr;
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devA), aBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devBAll), bBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devSlot), slotBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK(aclrtMemcpy(devA, aBytes, hostA.data(), aBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devBAll, bBytes, hostBAll.data(), bBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devSlot, slotBytes, hostSlot.data(), slotBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  constexpr int32_t c2vBuf = 0x10000;

  LaunchMatmulTPushPopLoop4Print(devA, devBAll, devSlot, c2vBuf, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::puts("Kernel finished. Expect 4 TPRINT blocks with 8x16 outputs filled "
            "with 1.0, 2.0, 3.0, 4.0 in order.");

  aclrtFree(devA);
  aclrtFree(devBAll);
  aclrtFree(devSlot);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return 0;
}
