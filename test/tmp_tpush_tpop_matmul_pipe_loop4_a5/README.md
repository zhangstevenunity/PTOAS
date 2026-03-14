# A5 临时样例: 4 次 TMATMUL + TPUSH/TPOP(UB) 顺序检查

这个目录是临时验证用例，可直接删除。

## 目标

- 复用 `initialize_l2l_pipe` 的 C2V 路径
- Cube 侧循环 4 次执行 `tpush`
- Vector 侧循环 4 次执行 `tpop`
- 检查 `tpop` 看到的 4 个输出块是否按 FIFO 顺序分别为全 `1.0`、全 `2.0`、全 `3.0`、全 `4.0`

## 数据构造

- `A = 16x16` 单位阵
- `B_all = 4 x (16x16)` 的连续 GM 数据
- 第 1/2/3/4 个 `16x16` 子块分别填满 `1.0 / 2.0 / 3.0 / 4.0`
- 因为 `A * B = I * B = B`，所以 `tpush` 入 FIFO 的结果块和 `B` 一致

## 目录说明

- `kernel.mlir`
  - PTOAS 源 IR
  - Cube: `tload(gm->mat)` -> `tmov(mat->left/right)` -> `tmatmul` -> `tpush(acc, pipe)`，循环 4 次
  - Vector: `tpop(pipe)` -> `get_fifo_tile(pipe, slot)` -> `tprint` -> `tfree`，循环 4 次
- `kernel_a5_manual.cpp`
  - 可运行的 A5 kernel C++，保留 `TPRINT`
- `main.cpp`
  - Host 侧 ACL 启动代码，构造 `A` 和 4 段 `B`

## 1. 用 PTOAS 检查 MLIR

在仓库根目录执行：

```bash
./build/tools/ptoas/ptoas --pto-arch=a5 test/tmp_tpush_tpop_matmul_pipe_loop4_a5/kernel.mlir > /tmp/tpush_tpop_matmul_pipe_loop4.out
```

说明：和现有临时样例一致，`kernel.mlir` 里的 `pto.tprint` 在部分 pass 组合下可能被提前折叠。
所以这个目录同时提供 `kernel_a5_manual.cpp`，用于保留运行路径上的 `TPRINT`，直接检查 4 次 `tpop` 输出。

## 2. 运行预期

运行 `kernel_a5_manual.cpp` + `main.cpp` 后，应看到 4 段 `TPRINT` 输出。

- 第 1 段：全 `1.0`
- 第 2 段：全 `2.0`
- 第 3 段：全 `3.0`
- 第 4 段：全 `4.0`

由于 A5 C2V 示例里的 consumer tile 是 `8x16` Vec tile，所以每次打印的是对应结果块的 `8x16` 输出。

## 3. 参数约定

- Tile 形状：`16x16`
- 输入类型：`f32`
- 输出累加类型：`f32`
- `pipe`：`initialize_l2l_pipe {dir_mask = 1}`，C2V，A5 下映射 UB `VEC_FIFO`
- `main.cpp` 里 `c2vBuf = 0x10000` 为示例 UB 基址
