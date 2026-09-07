<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# PTOAS `Layout::NZ` 推断与 pto-isa 标准五维形对齐

关联 issue: [#527](https://github.com/hw-native-sys/PTOAS/issues/527)。

本文定义 PR #1027 的最终实现约束。设计、实现、测试和文档都在同一个 PR 中完成；
PR 合入后关闭 #527。

范围仅包含 GM 侧 `GlobalTensor` 的 layout 解析与传播
（`tensor_view` / `partition_tensor_view` / memref view → `pto::Layout`），不改变 tile
侧 `blayout/slayout`，也不改变 fixpipe 的 `NZ2ND/NZ2DN/NZ2NZ` 转换语义。

## 1. pto-isa 的 NZ 五维标准形

pto-isa 使用 `TileShape2D` / `BaseShape2D` 表示二维 NZ tensor。常量
`C0_SIZE_BYTE = 32`、`FRACTAL_NZ_ROW = 16`，标准根视图为：

```text
C0     = 32 / sizeof(T)
shape  = [1, cols / C0, rows / 16, 16, C0]
stride = [rows * cols, rows * C0, 16 * C0, C0, 1]
```

五个维度的固定语义如下：

| 维度 | 含义 | 标准根视图取值 |
|---|---|---|
| `d0` | 二维场景的占位维 | `1` |
| `d1` | 列分块数 `n1` | `cols / C0` |
| `d2` | 行分块数 `m1` | `rows / 16` |
| `d3` | fractal 内行数 | `16` |
| `d4` | fractal 内列数 | `C0` |

例如 fp32 `(rows, cols) = (128, 64)` 时：

```text
shape  = [1, 8, 8, 16, 8]
stride = [8192, 1024, 128, 8, 1]
```

标准根视图满足：

```text
shape[3] == 16
shape[4] == C0
stride[4] == 1
stride[3] == C0
stride[2] == 16 * C0
stride[1] == shape[2] * stride[2]
stride[0] == shape[1] * stride[1]
```

这里的 element 指 PTOAS/pto-isa C++ 类型的存储元素。对于
`float4_e1m2x2_t` / `float4_e2m1x2_t`，一个存储元素是一字节 packed pair，
shape 的列数也按 packed pair 计数，因此 `sizeof(T) == 1`、`C0 == 32`。只有存储
字节数为 0，或 32 不能被存储字节数整除时，才不能形成 NZ C0。

## 2. 当前问题

PTOAS 当前在三处重复实现 layout 推断：

| 位置 | 用途 |
|---|---|
| `lib/PTO/Transforms/InferPTOLayout.cpp` | 给 view、memref view 和 load/store 附加 layout |
| `lib/PTO/IR/PTO.cpp` | verifier 获取逻辑 layout |
| `lib/PTO/Transforms/PTOToEmitC.cpp` | EmitC 缺少 layout 属性时兜底 |

当前 NZ 条件为：

```cpp
shape[2] == 16
shape[2] * shape[3] * elemBytes == 512
stride[4] == 1
stride[3] == shape[4]
```

它相对 pto-isa 标准形错位一维，并且没有验证 `stride[2..0]`。结果是：

- canonical `[1,8,8,16,8]` fp32 被判成 ND；
- `[4,1,16,8,16]` 这类连续 ND 形状被误判成 NZ；
- 显式 `{layout = #pto.layout<nz>}` 仍会被错误的 pattern 推断否决。

## 3. 必须保留的语义边界

canonical NZ 根视图的数值 stride 同时满足连续五维 ND 的累积积关系。因此仅凭
shape/stride 无法证明用户的逻辑语义一定是 NZ。

本 PR 采用一条确定的解析顺序：

1. **显式 layout 是权威来源。** 显式 `ND`、`DN`、`NZ` 都不会被 pattern 推断覆盖；
2. **派生 view 继承已经解析的源 layout。** NZ 子视图需要额外通过 fractal 边界校验；
3. **只有无显式属性、无源 layout 的根视图才走 pattern 推断；**
4. pattern 推断把满足 canonical NZ 规则的非退化二维五维形约定为 NZ，其他情况继续
   使用现有 ND/DN 推断。

第 4 条是一项 PTOAS 输入约定，不声称它能从数值上区分所有连续 5D ND。用户需要表达
同 shape/stride 的 ND 时，必须能够通过显式 `layout = nd` 覆盖该约定。

## 4. 唯一实现

### 4.1 共享 layout 工具

新增 `include/PTO/IR/PTOLayoutUtils.h` 和
`lib/PTO/IR/PTOLayoutUtils.cpp`，集中实现：

```cpp
namespace mlir::pto {

std::optional<std::string>
getNZViewCompatibilityError(ArrayRef<int64_t> shape5D,
                            ArrayRef<int64_t> stride5D,
                            unsigned storageElemBytes);

std::optional<std::string>
getNZSubviewCompatibilityError(ArrayRef<int64_t> sourceShape5D,
                               ArrayRef<int64_t> offsets5D,
                               ArrayRef<int64_t> sizes5D);

bool isLayoutCompatible5D(Layout layout,
                          ArrayRef<int64_t> shape,
                          ArrayRef<int64_t> stride,
                          unsigned storageElemBytes);

std::optional<Layout>
inferLayout5D(ArrayRef<int64_t> shape5D,
              ArrayRef<int64_t> stride5D,
              unsigned storageElemBytes,
              std::optional<Layout> preferredMinor2D = std::nullopt,
              bool *isMinor2DAmbiguous = nullptr);

} // namespace mlir::pto
```

`getNZC0StorageElems()`、`isNZViewCompatible5D()` 和
`isCanonicalNZRoot5D()` 仅是该实现文件内部的组合谓词，不作为跨 pass API 导出。

`InferPTOLayout.cpp`、`PTO.cpp`、`PTOToEmitC.cpp` 都调用这套实现，删除各自的私有
NZ/ND/DN 判定副本。调用点必须先解析显式属性和源 view 属性，只有确实缺失 layout 时
才调用 `inferLayout5D()`。

### 4.2 根视图与子视图使用不同谓词

`isCanonicalNZRoot5D()` 只用于无 layout 根视图的 pattern 推断，要求完整的紧密
canonical stride：

```cpp
NZInnerStructure(shape, stride, bytes)
    && stride[1] == shape[2] * stride[2]
    && stride[0] == shape[1] * stride[1]
```

其中 `NZInnerStructure` 是下列不能被切开的内部 fractal 条件的伪代码：

```cpp
shape[3] == 16
shape[4] == C0
stride[4] == 1
stride[3] == C0
stride[2] == 16 * C0
```

`isNZViewCompatible5D()` 在内部结构之上验证当前 view 的外层跨度：

```cpp
NZInnerStructure(shape, stride, bytes)
    && shape[0] == 1
    && stride[1] >= shape[2] * stride[2]
    && stride[1] % C0 == 0
    && stride[0] >= shape[1] * stride[1]
    && stride[0] % C0 == 0
```

它允许 `stride[0]` / `stride[1]` 包含按 C0 对齐的外层 gap，但不允许负 gap、重叠
或破坏 32B block 对齐。`isCanonicalNZRoot5D()` 等价于
`isNZViewCompatible5D()` 再要求两个外层 stride 都等于当前 shape 的紧密跨度。

不能使用根视图的外层紧密 stride 等式验证 partition 结果。合法子视图缩小 `d1`
或 `d2` 后会保留父 stride，此时 `stride[0]` 或 `stride[1]` 可以大于当前子视图的
紧密跨度；pto-isa 的 NZ load/store 使用这个差值表示外层 gap。

### 4.3 无属性根视图的 NZ 约定

无显式 layout 且没有可继承源 layout 时，按以下唯一规则推断 NZ：

```cpp
isCanonicalNZRoot5D(shape, stride, storageElemBytes)
    && shape[0] == 1
    && (shape[1] > 1 || shape[2] > 1)
```

`[1,1,1,16,C0]` 单 fractal 不自动升级为 NZ，因为它是 rank2
`[16,C0]` ND 视图规范化后的常见形状。用户确实需要单 fractal NZ 时使用显式
`layout = nz`。

显式 `layout = nd` 必须覆盖上述 pattern 约定。例如 fp32：

```text
shape  = [1, 2, 1, 16, 8]
stride = [256, 128, 128, 8, 1]
layout = nd
```

该 shape/stride 同时符合 canonical NZ 数值形式和连续 5D ND，PTOAS 必须保留用户
声明的 ND，不得报 layout mismatch。

### 4.4 显式 layout 校验

显式属性不再与 pattern 推断结果比较，而是按其自身语义校验：

- 显式 `NZ` 根视图必须满足 NZ 内部 fractal 结构；canonical 紧密 stride 和合法的
  外层 block gap 都可接受，统一使用 `isNZViewCompatible5D()` 校验；
- 显式 `ND` / `DN` 保持为逻辑布局的权威声明。它们不承诺当前 view 是紧密
  minor-2D 存储，因此不使用 dense stride 递推拒绝已有的 interleaved/gapped
  view；minor-2D 歧义处理只用于无显式 layout 的推断；
- 校验失败时报告具体的 shape/stride 不变量，不回退到其他 layout；
- 显式属性不附加 `pto.inferred_layout`。

### 4.5 NZ partition/subview 传播

源 layout 为 NZ 时，`partition_view` 和 `memref.subview` 按源 view 校验：

- `d3`、`d4` 必须完整保留：offset 为 0，size 等于源 size；
- `d1`、`d2` 可以缩小，结果保留源 stride；
- `d1`、`d2` 的一个坐标单位本身就是一个完整 block，因此任意合法整数 offset
  都是 block 边界，不再增加含义不清的二次对齐条件；
- 二维 NZ 的 `d0` 保持 `offset = 0`、`size = 1`；
- 校验通过后直接继承 NZ，不再用 `isCanonicalNZRoot5D()` 重新推断；
- `partition_view` 可以继续以 `partition_tensor_view` 为源，嵌套切分逐级沿源链
  继承同一个 layout；lowering 必须同时保留根 view 的真实 stride 和累计 offset；
- EmitC 无法从源类型或源 view 链解析精确 stride 时必须报错，不能用连续
  row-major stride 猜测；
- 在 `d3`/`d4` 内部切分时发出错误，不静默改成 ND。

动态 `d1`/`d2` offset 和 size 不改变 fractal 内部结构，可以传播 NZ。`d3`/`d4`
是否完整必须能够在编译期证明，否则报错。

### 4.6 动态 shape/stride

显式或继承的 layout 在动态 shape 下继续携带。校验所有编译期已知的结构维，不使用
未知值否决显式 NZ。

无属性根视图只有在 NZ 内部结构和非退化门槛都能在编译期证明时才自动推断 NZ；无法
证明时不猜测，沿用 ND/DN fallback，调用方应提供显式 layout。

### 4.7 旧规则处理

直接删除旧的错位 NZ 判定，只保留本设计定义的标准规则。不保留旧规则并集，不增加
legacy 开关。旧规则命中的非标准形属于误判，继续兼容会固定错误的
`GlobalTensor<..., Layout::NZ>` 模板参数。

### 4.8 控制流和函数边界传播

layout 传播使用模块级等价关系，并按以下 lattice 合并：

```text
Unknown
ND / DN / NZ / MX_A_ZZ / MX_B_NN
Conflict
```

`partition_view`、`arith.select`、`scf.if`、`scf.for`、`scf.while`、
`scf.execute_region`、`scf.index_switch`、CF block argument、直接
`func.call` 和 `func.return` 都建立 layout 等价约束。两个已知且不同的 layout 进入
同一等价集合时立即报告 conflict；不能在 merge 后默认成 ND。

`TensorViewType` 和 `PartitionTensorViewType` 增加可选 layout 参数。无参数形式保持原有
文本兼容，表示尚未解析或默认 ND；非 ND 解析结果写入类型，例如：

```mlir
!pto.tensor_view<1x8x8x16x8xf32, #pto.layout<nz>>
!pto.partition_tensor_view<1x4x8x16x8xf32, #pto.layout<nz>>
```

因此结构化控制流转换和函数签名转换可以直接从类型得到 layout，不再依赖 defining-op
回溯。direct internal helper 的参数、返回值和所有调用点必须形成一致契约；外部声明按
其显式类型 layout 解析，未写 layout 时使用 ND。view-typed indirect call 因无法静态
解析 callee 契约而拒绝。

## 5. 示例

以下示例省略 `arith.constant` 定义，常量名与数值一致。

### 5.1 canonical NZ 自动推断

输入是无显式 layout 的 fp32 canonical NZ 根视图：

```mlir
%view = pto.make_tensor_view %dst,
  shape = [%c1, %c8, %c8, %c16, %c8],
  strides = [%c8192, %c1024, %c128, %c8, %c1]
  : !pto.tensor_view<1x8x8x16x8xf32>

%part = pto.partition_view %view,
  offsets = [%c0, %c0, %c0, %c0, %c0],
  sizes = [%c1, %c8, %c8, %c16, %c8]
  : !pto.tensor_view<1x8x8x16x8xf32>
    -> !pto.partition_tensor_view<1x8x8x16x8xf32>
```

解析顺序中没有显式属性和源 layout，因此根视图进入 pattern 推断。它满足
`isCanonicalNZRoot5D()` 和非退化门槛，预期生成：

```cpp
GlobalTensor<
    float,
    pto::Shape<1, 8, 8, 16, 8>,
    pto::Stride<8192, 1024, 128, 8, 1>,
    pto::Layout::NZ>
```

`make_tensor_view` 会得到 `layout = #pto.layout<nz>` 和
`pto.inferred_layout = true`，full-size partition 继承 NZ。

### 5.2 相同数值形式由显式 ND 覆盖

下面的连续 5D ND 在 shape/stride 数值上也满足 canonical NZ pattern，但用户显式
声明其逻辑语义为 ND：

```mlir
%view = pto.make_tensor_view %dst,
  shape = [%c1, %c2, %c1, %c16, %c8],
  strides = [%c256, %c128, %c128, %c8, %c1]
  {layout = #pto.layout<nd>}
  : !pto.tensor_view<1x2x1x16x8xf32>
```

显式属性在 pattern 推断之前解析，预期生成：

```cpp
GlobalTensor<
    float,
    pto::Shape<1, 2, 1, 16, 8>,
    pto::Stride<256, 128, 128, 8, 1>,
    pto::Layout::ND>
```

该 view 不附加 `pto.inferred_layout`，也不得出现
`layout mismatch: user-specified layout=nd but inferred=nz`。

### 5.3 NZ partition 的合法和非法切分

以 5.1 的 `%view` 为源，在 `d1` 上从第 2 个列块开始取 4 个块：

```mlir
%d1_part = pto.partition_view %view,
  offsets = [%c0, %c2, %c0, %c0, %c0],
  sizes = [%c1, %c4, %c8, %c16, %c8]
  : !pto.tensor_view<1x8x8x16x8xf32>
    -> !pto.partition_tensor_view<1x4x8x16x8xf32>
```

结果基址前移 `2 * stride[1] = 2048` 个 fp32 storage element，保留父 stride：

```cpp
GlobalTensor<
    float,
    pto::Shape<1, 4, 8, 16, 8>,
    pto::Stride<8192, 1024, 128, 8, 1>,
    pto::Layout::NZ>
```

此时 `stride[0] != shape[1] * stride[1]`，但这是合法的外层 gap，不能用 canonical
根视图等式把它拒绝或重新推断成 ND。

同理，在 `d2` 上取 4 个行块也合法：

```mlir
%d2_part = pto.partition_view %view,
  offsets = [%c0, %c0, %c2, %c0, %c0],
  sizes = [%c1, %c8, %c4, %c16, %c8]
  : !pto.tensor_view<1x8x8x16x8xf32>
    -> !pto.partition_tensor_view<1x8x4x16x8xf32>
```

它的基址前移 `2 * stride[2] = 256` 个 storage element，stride 仍为
`[8192,1024,128,8,1]`，layout 继续是 NZ。

在 `d3` 或 `d4` 内切分会破坏一个完整 fractal。例如：

```mlir
%bad = pto.partition_view %view,
  offsets = [%c0, %c0, %c0, %c1, %c0],
  sizes = [%c1, %c8, %c8, %c15, %c8]
  : !pto.tensor_view<1x8x8x16x8xf32>
    -> !pto.partition_tensor_view<1x8x8x15x8xf32>
```

预期在 verifier/layout pass 阶段失败，不生成 ND fallback：

```text
error: NZ view cannot be partitioned inside a fractal:
       d3 must keep offset=0 and size=16 (got offset=1, size=15)
```

## 6. 影响面

对 `test/lit/pto/*.pto` 和 `test/lit/tile_fusion/*.pto` 的现有样本扫描得到：

- 242 个文件生成 1558 个 `GlobalTensor` 实例化点；
- 当前 NZ 去重形状中，两组符合 pto-isa 标准形；
- 两组只存在于 `globaltensor_layout_bytewidth_emitc.pto` 的旧错位形状不再是 NZ：
  int8 样例回到 ND；末维为 1 时 ND/DN 的寻址等价，未提供消费者偏好时保持
  兼容默认 ND；
- 不加非退化门槛时，38 个文件中的 80 个单-fractal ND 视图会被升级成 NZ；
- 加门槛后，当前语料没有新增 NZ。

“当前语料没有新增 NZ”仅是回归影响数据，不代表 shape/stride 已经从理论上完成 ND/NZ
消歧。因此实现仍必须保证所有显式 layout 优先，并增加“canonical 数值形式 +
显式 ND”的反例测试。

`mgather`/`mscatter` 的 ND-only 校验也必须使用同一解析顺序：显式/继承 layout
优先，缺失时才 pattern 推断。真实 NZ 仍按现有约束报错；显式 ND 不得被 heuristic
改成 NZ。

## 7. 本 PR 的实现范围

PR #1027 在现有设计提交之后继续完成以下内容：

- 新增共享 `PTOLayoutUtils`；
- 同步 IR verifier、layout pass 和 EmitC fallback；
- 实现显式 layout 全优先；
- 替换旧 NZ 规则；
- 实现 NZ partition/subview 的 source-relative 校验和传播；
- 实现跨 SCF/CF merge 与函数参数、返回值、调用点的 layout 等价传播和冲突诊断；
- 在 view 类型中携带已解析的非 ND layout，供结构化转换和 EmitC ABI 使用；
- 补齐静态、动态、packed storage element 和跨调用点一致性测试；
- 更新用户文档及 Python `make_tensor_view` 的显式 layout 示例。

不再拆分成多个后续 PR。上述实现和验证全部完成后将 #1027 转为 ready；合入该 PR
即关闭 #527。

## 8. 测试要求

新增 `test/lit/pto/issue527_nz_canonical_view_infer.pto`，至少覆盖：

1. fp32 canonical `[1,8,8,16,8]` 自动推断 NZ；
2. half `[1,4,8,16,16]`、int8 `[1,2,8,16,32]` 自动推断 NZ；
3. 旧错位形状 `[4,1,16,8,16]` 推断 ND；
4. 单 fractal `[1,1,1,16,C0]` 无属性时保持 ND；
5. 单 fractal 显式 NZ 时保留 NZ；
6. canonical 数值形式 `[1,2,1,16,8]` 显式 ND 时保留 ND；
7. canonical NZ 显式 NZ 时不再出现 layout mismatch；
8. 非法 NZ 内部 shape/stride 显式 NZ 时给出精确诊断；
9. `d1`、`d2` 分别缩小的 partition 保留父 stride 并继承 NZ；
10. `d3` 或 `d4` 内部切分时报错；
11. 嵌套 partition 继续从已经解析的源 layout 传播；
12. FP4 packed-pair 按一字节 storage element 计算 C0；
13. verifier、pass 和 EmitC 对同一 view 得到一致 layout；
14. mgather/mscatter 的 ND-only 路径不覆盖显式 ND；
15. 带外层 gap 的根 view 经过嵌套 partition 后仍保留原 stride，并累计两级 offset；
16. `scf.if(NZ, NZ)` 经过 helper、`scf.for`、`scf.while` 后仍生成 NZ；
17. `scf.if(NZ, ND)` 在 merge 点报告 layout conflict；
18. direct helper 的参数与返回值使用同一 NZ `GlobalTensor` ABI。

同时更新 `globaltensor_layout_bytewidth_emitc.pto` 中固化旧错位规则的期望值，并增加
A3/A5 编译覆盖。具备板卡资源时运行一条 NZ load/store E2E 数据比对，确认生成的
`GlobalTensor` shape、stride、layout 与 pto-isa 行为一致。
