# A5 临时样例: TMATMUL + TPUSH/TPOP(UB) + TPRINT

这个目录是临时验证用例，可直接删除。

## 目录说明

- `kernel.mlir`
  - PTOAS 源 IR，表达目标链路：
    - Cube: `tload(gm->mat)` -> `tmov(mat->left/right)` -> `tmatmul` -> `tpush(acc, pipe)`
    - Vector: `tpop(vec, pipe)` -> `tprint`
- `kernel_a5_manual.cpp`
  - 可运行的 A5 kernel C++（保留 `TPRINT`）
- `main.cpp`
  - Host 侧 ACL 启动代码，构造输入并拉起 kernel
  - 输入设为 `A=单位阵`、`B=全1矩阵`，便于按 `TPRINT` 结果快速验算

## 1. 用 PTOAS 检查 MLIR

在仓库根目录执行：

```bash
./build/tools/ptoas/ptoas --pto-arch=a5 test/tmp_tpush_tpop_matmul_pipe_a5/kernel.mlir > /tmp/tpush_tpop_matmul_pipe.out
```

可提取 emitc 末尾 C++：

```bash
awk 'BEGIN{emit=0} /^#include "pto\/pto-inst.hpp"/{emit=1} emit{print}' /tmp/tpush_tpop_matmul_pipe.out > /tmp/tpush_tpop_matmul_pipe_emitc.cpp
```

说明：当前流水线中，`kernel.mlir` 里的 `pto.tprint` 在部分 pass 组合下可能被提前折叠。
所以本目录提供了 `kernel_a5_manual.cpp` 来保证运行路径上确实有 `TPRINT`。

## 2. 运行方式（参考 pto-isa A5 st 框架）

将 `kernel_a5_manual.cpp` + `main.cpp` 按你现有 A5 st 工程的 CMake 方式接入并编译。

运行后，Vector 段执行 `TPRINT(vecOut)`，日志里应可看到打印输出。
本样例里理论结果应是全 `1.0`（`A * B = I * Ones = Ones`）。

## 3. 参数约定

- Tile 形状：`16x16`
- 输入类型：`f32`
- 输出累加类型：`f32`
- `pipe`：`dir_mask = 1`，C2V，A5 下映射 UB `VEC_FIFO`
- `main.cpp` 里 `c2vBuf = 0x10000` 为示例 UB 基址
