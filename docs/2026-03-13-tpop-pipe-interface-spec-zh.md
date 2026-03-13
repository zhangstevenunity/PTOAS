# PTOAS Pipe & TPUSH/TPOP 接口定义

---

## 1. 概述

PTOAS 提供跨 Cube/Vector 单元的 pipe 通信机制，用于 Cube 和 Vector 之间的 tile 数据传递。根据数据流转路径，定义两种 pipe 初始化指令：

| 指令 | 数据路径 | 含义 |
|---|---|---|
| `pto.initialize_l2g2l_pipe` | local → GM → local | 数据经 GM 中转，适用于 A3 和 A5 |
| `pto.initialize_l2l_pipe` | local → local | 数据在 local buffer 内直传，仅 A5 |

两种 pipe 共享统一的操作接口——生产者：`pto.tpush`；消费者：`pto.tpop` → `pto.get_fifo_tile` → `pto.tfree`。

其中，`pto.tfree(%pipe, %slot_id)` 是**必写**的资源释放操作：

- `pto.tpop` 借出一个 slot
- `pto.get_fifo_tile` 暴露该 slot 对应的 tile 视图
- 用户在 tile 使用结束后，必须显式写出对应的 `pto.tfree`
- 编译器会对这组接口的配对关系和使用顺序做 verify 检查

### 编译变换过程

```
前端 IR                            Lowered IR                      EmitC
────────                           ──────────                      ─────
pto.initialize_l2g2l_pipe(...)  ─→                              ─→ TPipe<..., GM_FIFO, ...>(gm, local)
pto.initialize_l2l_pipe(...)    ─→                              ─→ TPipe<..., VEC/MAT_FIFO, ...>(local)

pto.tpush(%tile, %pipe)         ─→                              ─→ TPUSH(tile, pipe)

pto.tpop(%pipe) → %slot_id     ┐
                                ├→ pto.tpop_internal(%tile, %pipe) → TPOP(tile, pipe)
pto.get_fifo_tile(%pipe, %sid) ┘       [DPS]

pto.tfree(%pipe, %slot_id)     ─→ pto.tfree_internal(%pipe)     ─→ TFREE(pipe)
```

---

## 2. Pipe 初始化指令

### 2.1 `pto.initialize_l2g2l_pipe`

创建经 GM 中转的 pipe。生产者将 tile 写入 GM，消费者 TPOP 时从 GM 搬运到 local FIFO slot，再暴露给用户。

**数据路径：** `local(producer) → GM FIFO → local FIFO(consumer)`

**语法：**

```mlir
%pipe = pto.initialize_l2g2l_pipe {
    dir_mask = <i8>,
    local_fifo_depth = <i8>           // 可选，默认 2
}
    ( <gm_addr> : memref<..., #pto.address_space<gm>>
      [, <local_addr> : i32] )
    -> !pto.pipe<SrcTileType, DstTileType>
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `dir_mask` | `i8`（属性） | 方向：1 = C2V（Cube→Vector），2 = V2C（Vector→Cube） |
| `local_fifo_depth` | `i8`（属性，可选） | local FIFO 的 slot 深度，默认 2（double-buffering） |
| `gm_addr` | `memref<..., #pto.address_space<gm>>`（操作数，必须） | GM FIFO 基地址 |
| `local_addr` | `i32`（操作数，可选） | local FIFO 基地址。省略时由 plan memory 分配 |

**`local_fifo_depth` 说明：**
- 指定 TPOP 从 GM 搬运数据到 local buffer 时，local FIFO 的 slot 深度
- plan memory / analysis pass 根据此值计算 local buffer 分配量（= `local_fifo_depth × slot_size`）
- `slot_size = max(srcTile.size, dstTile.size)`（见第 9 节）
- 默认值 2，支持 double-buffering

**结果：** `!pto.pipe<SrcTileType, DstTileType>`

**校验规则：**
- `dir_mask` 必须为 1 或 2
- `local_fifo_depth` 如指定须大于 0
- `gm_addr` 必须带 GM address space
- `local_addr` 如存在须为 `i32`

**可用架构：** A3、A5

**示例：**

```mlir
// 指定 GM 地址和 local 地址
%pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
    (%gm_addr : memref<64x128xf32, #pto.address_space<gm>>, %local_addr : i32)
    -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>,
                 memref<32x128xf32, #pto.address_space<vec>>>

// 省略 local 地址，由 plan memory 分配
%pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
    (%gm_addr : memref<64x128xf32, #pto.address_space<gm>>)
    -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>,
                 memref<32x128xf32, #pto.address_space<vec>>>

// 指定 local_fifo_depth=4
%pipe = pto.initialize_l2g2l_pipe {dir_mask = 1, local_fifo_depth = 4}
    (%gm_addr : memref<64x128xf32, #pto.address_space<gm>>, %local_addr : i32)
    -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>,
                 memref<32x128xf32, #pto.address_space<vec>>>
```

---

### 2.2 `pto.initialize_l2l_pipe`

创建 local 直传 pipe。生产者将 tile 写入 local FIFO slot，消费者 TPOP 时直接暴露该 slot 给用户。数据不经过 GM。

**数据路径：** `local(producer) → local FIFO(consumer)`

**语法：**

```mlir
%pipe = pto.initialize_l2l_pipe {
    dir_mask = <i8>
}
    ( [<local_addr> : i32] )
    -> !pto.pipe<SrcTileType, DstTileType>
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `dir_mask` | `i8`（属性） | 方向：1 = C2V（Cube→Vector），2 = V2C（Vector→Cube） |
| `local_addr` | `i32`（操作数，可选） | local FIFO 基地址。省略时由 plan memory 分配 |

**结果：** `!pto.pipe<SrcTileType, DstTileType>`

**校验规则：**
- `dir_mask` 必须为 1 或 2
- `local_addr` 如存在须为 `i32`

**可用架构：** 仅 A5

**示例：**

```mlir
// C2V，指定 local 地址
%pipe = pto.initialize_l2l_pipe {dir_mask = 1}
    (%local_addr : i32)
    -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>,
                 memref<32x128xf32, #pto.address_space<vec>>>

// V2C，省略地址由 plan memory 分配
%pipe = pto.initialize_l2l_pipe {dir_mask = 2}
    -> !pto.pipe<memref<32x128xf32, #pto.address_space<vec>>,
                 memref<64x128xf32, #pto.address_space<mat>>>
```

---

### 2.3 两种 Pipe 的对比

| 特性 | `initialize_l2g2l_pipe` | `initialize_l2l_pipe` |
|---|---|---|
| 数据路径 | local → GM → local | local → local |
| FIFOType | `GM_FIFO` | `VEC_FIFO`（C2V）/ `MAT_FIFO`（V2C） |
| 可用架构 | A3 + A5 | 仅 A5 |
| GM 地址 | 必须 | 无 |
| local 地址 | 可选（plan memory 可分配） | 可选（plan memory 可分配） |
| `local_fifo_depth` | 可选，默认 2 | 不适用（深度 = FiFoDepth） |
| TPOP 行为 | wait → DMA(GM→local) → bind tile | wait → bind tile |
| TPOP 流水线 | `PIPE_MTE2`（涉及 DMA） | `PIPE_S`（仅地址赋值） |

---

## 3. 生产者/消费者指令

### 3.1 `pto.tpop`

等待 pipe 消费者 slot 就绪，返回 slot ID。

**语法：**

```mlir
%slot_id = pto.tpop ( %pipe : !pto.pipe<S, D> ) -> index
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `pipe_handle` | `!pto.pipe<S, D>` | 待消费的 pipe |

**结果：** `index` — 获取到的 ring buffer slot ID。

**Traits / Interfaces：**
- 不实现 `OpPipeInterface`（lowering 后由 `tpop_internal` 承载流水线信息）
- `MemoryEffectsOpInterface`：`pipe_handle` Read + Write

**`%slot_id` 使用约束：**
- 只能被 `pto.get_fifo_tile` 和 `pto.tfree` 使用
- 必须有且仅有一个 `pto.get_fifo_tile` 消费
- 必须有且仅有一个 `pto.tfree` 消费
- `pto.get_fifo_tile` 和 `pto.tfree` 必须使用与 `pto.tpop` 相同的 `pipe_handle`
- 当前实现要求对应的 `pto.get_fifo_tile` 和 `pto.tfree` 与该 `pto.tpop` 位于同一个 block 中

---

### 3.2 `pto.get_fifo_tile`

将 slot ID 解析为指向对应 local FIFO entry 的 tile 视图。纯地址计算，不搬运数据。

**语法：**

```mlir
%tile = pto.get_fifo_tile ( %pipe, %slot_id : !pto.pipe<S, D>, index )
    -> DstTileType
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `pipe_handle` | `!pto.pipe<S, D>` | 拥有 FIFO 的 pipe |
| `slot_id` | `index` | 由 `pto.tpop` 返回的 slot ID |

**结果：** `DstTileType` — 从 `pipe.dstTileType` 推导。结果 tile 指向 FIFO slot 在 local buffer 中的存储。

**Traits / Interfaces：**
- `ViewLikeOpInterface`
- `MemoryEffectsOpInterface`：`pipe_handle` Read；结果无 Allocate

**`%tile` 使用约束：**
- 只读：可作为下游指令的源操作数（`ins`），不可作为目标操作数（`outs`）
- 不可在对应的 `pto.tfree` / `pto.tfree_internal` 之后使用

---

### 3.3 `pto.tfree`

释放 FIFO slot。必写；每个 `pto.tpop` 都必须显式对应一个 `pto.tfree`。

**语法：**

```mlir
pto.tfree ( %pipe, %slot_id : !pto.pipe<S, D>, index )
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `pipe_handle` | `!pto.pipe<S, D>` | 拥有该 slot 的 pipe |
| `slot_id` | `index` | 待释放的 slot ID |

**Traits / Interfaces：**
- 不实现 `OpPipeInterface`
- `MemoryEffectsOpInterface`：`pipe_handle` Read + Write

**使用约束：**
- 必须与对应 `pto.tpop` 的 `pipe_handle` 一致
- 当前实现要求位于与对应 `pto.tpop` 相同的 block 中
- 必须出现在对应 `pto.get_fifo_tile` 之后
- 必须晚于该 borrowed tile 的所有使用

---

### 3.4 `pto.tpush`

生产者侧推送。生产者写入 pipe 的后备存储（l2g2l_pipe 写 GM，l2l_pipe 写 local buffer）。

```mlir
pto.tpush(%src_tile, %pipe : SrcTileType, !pto.pipe<SrcTileType, DstTileType>)
```

---

## 4. 使用示例

### 4.1 L2L Pipe（A5 local 直传）

```mlir
func.func @c2v_l2l_example(%local_addr: i32) {
  %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%local_addr : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>,
                   memref<32x128xf32, #pto.address_space<vec>>>

  pto.section.cube {
    pto.tpush(%acc_tile, %pipe : ...)
  }

  pto.section.vector {
    %slot_id = pto.tpop(%pipe : !pto.pipe<...>) -> index
    %tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<...>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>
    pto.tmov ins(%tile : ...) outs(%dst : ...)
    pto.tfree(%pipe, %slot_id : !pto.pipe<...>, index)
  }
  return
}
```

### 4.2 L2G2L Pipe（经 GM 中转）

```mlir
func.func @c2v_l2g2l_example(%gm_addr: memref<64x128xf32, #pto.address_space<gm>>,
                              %local_addr: i32) {
  %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
      (%gm_addr : memref<64x128xf32, #pto.address_space<gm>>, %local_addr : i32)
      -> !pto.pipe<memref<64x128xf32, #pto.address_space<acc>>,
                   memref<32x128xf32, #pto.address_space<vec>>>

  pto.section.cube {
    pto.tpush(%acc_tile, %pipe : ...)
  }

  pto.section.vector {
    %slot_id = pto.tpop(%pipe : !pto.pipe<...>) -> index
    %tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<...>, index)
        -> memref<32x128xf32, #pto.address_space<vec>>
    pto.tmov ins(%tile : ...) outs(%dst : ...)
    pto.tfree(%pipe, %slot_id : !pto.pipe<...>, index)
  }
  return
}
```

---

## 5. 内部 Op（Lowered IR）

以下 Op 仅在 `LowerTPop` pass 后出现，不属于前端 IR。

### 5.1 `pto.declare_tile`

声明未绑定地址的 tile，地址在运行时由 `pto.tpop_internal` 赋值。

```mlir
%tile = pto.declare_tile -> TileBufType
```

### 5.2 `pto.tpop_internal`

统一的 TPOP，DPS 格式。等待 slot 就绪，绑定 tile 地址到 FIFO slot。l2g2l pipe 还会执行 GM → local 搬运。

```mlir
pto.tpop_internal ( %tile, %pipe : TileBufType, !pto.pipe<S, D> )
    { assigned_pipe = #pto.pipe<PIPE_xxx> }
```

| pipe 类型 | assigned_pipe | 原因 |
|---|---|---|
| l2g2l_pipe（GM_FIFO） | `PIPE_MTE2` | 涉及 GM → local DMA 搬运 |
| l2l_pipe（VEC/MAT_FIFO） | `PIPE_S` | 仅地址赋值 |

**运行时语义：**

```
1. wait_until(pipe.consumer_slot_ready())
2. [l2g2l_pipe] dma_copy(pipe.gm_slot_addr() → pipe.local_slot_addr())
3. tile.rebind_addr(pipe.local_slot_addr())
4. pipe.advance_consumer_cursor()
```

### 5.3 `pto.tfree_internal`

释放 pipe slot。由显式 frontend `pto.tfree` lowering 生成。

```mlir
pto.tfree_internal ( %pipe : !pto.pipe<S, D> )
```

- `OpPipeInterface`：`PIPE_S`（scalar 操作）
- EmitC：`TFREE(pipe)`（A3 和 A5 均生成，因为都使用 local FIFO）

---

## 6. 编译变换与校验

### 6.1 `LowerTPop` Pass

将前端 `tpop` + `get_fifo_tile` 融合为 `tpop_internal`，并将显式 `tfree` 降级为 `tfree_internal`。

**融合 Pattern：**

```
输入:                                     输出:
%sid = pto.tpop(%pipe)                    %tile = pto.declare_tile -> DstTileType
%tile = pto.get_fifo_tile(%pipe, %sid)    pto.tpop_internal(%tile, %pipe)
                                              { assigned_pipe = ... }
```

`assigned_pipe` 根据 pipe 的定义 op 确定：
- `initialize_l2g2l_pipe` → `PIPE_MTE2`
- `initialize_l2l_pipe` → `PIPE_S`

**降级 Pattern：**

```
输入:                                     输出:
pto.tfree(%pipe, %slot_id)                pto.tfree_internal(%pipe)
```

在 lowering 之前，编译器会先验证：

- 每个 `pto.tpop` 必须且只能有一个 `pto.get_fifo_tile`
- 每个 `pto.tpop` 必须且只能有一个显式 `pto.tfree`
- `pto.get_fifo_tile` / `pto.tfree` 必须与对应 `pto.tpop` 使用相同 `pipe_handle`
- 当前实现要求三者位于同一个 block 中，且 `pto.tfree` 出现在 `pto.get_fifo_tile` 之后

### 6.2 `VerifyTFree` Pass

对 lowered IR 中的 `tpop_internal` / `tfree_internal` 配对关系做合法性检查。

```
对每个 pto.tpop_internal:
  1. 要求后续同 block 中存在匹配的 tfree_internal
  2. 验证 borrowed tile 不会在 tfree_internal 之后继续使用
  3. 验证同一 pipe 在匹配 free 之前不会再次 tpop_internal
```

也就是说，用户负责显式写出 `pto.tfree`，编译器负责检查释放是否存在、是否足够晚、以及是否满足当前的单 outstanding 约束。

### 6.3 Pass Pipeline

```
前端 IR
  │
  ▼
LowerTPop
  │  - tpop + get_fifo_tile → declare_tile + tpop_internal
  │  - tfree → tfree_internal
  ▼
VerifyTFree
  │  - 验证显式 tfree_internal 的位置与 outstanding-pop 约束
  ▼
LoweringSyncToPipe
  │  - 高层 sync op → 低层 pipe sync op
  ▼
PTOInsertSync（可选）
  │  - 自动同步插入
  ▼
...（其他 transform）...
  ▼
PTOToEmitC
  │  - initialize_l2g2l_pipe → TPipe<..., GM_FIFO, ...>(gm, local)
  │  - initialize_l2l_pipe   → TPipe<..., VEC/MAT_FIFO, ...>(local)
  │  - declare_tile          → Tile<...> varname;
  │  - tpop_internal         → TPOP(tile, pipe);
  │  - tfree_internal        → TFREE(pipe)
  ▼
C++ 输出
```

---

## 7. EmitC Lowering

### 7.1 Pipe 初始化 → TPipe

**`initialize_l2l_pipe`（A5）：**

```mlir
%pipe = pto.initialize_l2l_pipe {dir_mask = 1} (%local_addr : i32)
```

```cpp
auto v28 = TPipe<0, FIFOType::VEC_FIFO, 8, 8, SrcTile, DstTile>(local_addr);
```

**`initialize_l2g2l_pipe`（A3 或 A5）：**

```mlir
%pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
    (%gm_addr : memref<...>, %local_addr : i32)
```

```cpp
auto v28 = TPipe<0, FIFOType::GM_FIFO, 8, 8, SrcTile, DstTile>(gm_addr, local_addr);
```

**`initialize_l2g2l_pipe`（指定 local_fifo_depth=4）：**

```mlir
%pipe = pto.initialize_l2g2l_pipe {dir_mask = 1, local_fifo_depth = 4}
    (%gm_addr : memref<...>, %local_addr : i32)
```

```cpp
auto v28 = TPipe<0, FIFOType::GM_FIFO, 8, 8, SrcTile, DstTile,
                 false, 4>(gm_addr, local_addr);
```

**FIFOType 映射：**

| Op | dir_mask | FIFOType |
|---|---|---|
| `initialize_l2l_pipe` | 1 (C2V) | `VEC_FIFO` |
| `initialize_l2l_pipe` | 2 (V2C) | `MAT_FIFO` |
| `initialize_l2g2l_pipe` | 1 (C2V) | `GM_FIFO` |
| `initialize_l2g2l_pipe` | 2 (V2C) | `GM_FIFO` |

### 7.2 其他 Op

| Op | EmitC 输出 |
|---|---|
| `declare_tile` | `Tile<...> varname;` |
| `tpop_internal` | `TPOP(tile, pipe);` |
| `tfree_internal` | `TFREE(pipe);` |

---

## 8. Op 定义汇总

### 前端 Op

| Op | 操作数 | 结果 | Pipeline | 可见性 |
|---|---|---|---|---|
| `pto.initialize_l2g2l_pipe` | `gm_addr [, local_addr]` + 属性 | `!pto.pipe<S, D>` | — | 前端 |
| `pto.initialize_l2l_pipe` | `[local_addr]` + 属性 | `!pto.pipe<S, D>` | — | 前端 |
| `pto.tpop` | `pipe` | `index`（slot_id） | — | 前端 |
| `pto.get_fifo_tile` | `pipe, slot_id` | `DstTileType` | — | 前端 |
| `pto.tfree` | `pipe, slot_id` | — | — | 前端 |
| `pto.tpush` | `tile, pipe` | — | — | 前端 |

### 内部 Op

| Op | 操作数 | 结果 | DPS | Pipeline |
|---|---|---|---|---|
| `pto.declare_tile` | — | `TileBufType` | 否 | — |
| `pto.tpop_internal` | `tile, pipe` + `assigned_pipe` | — | 是 | GM_FIFO: `PIPE_MTE2`；VEC/MAT_FIFO: `PIPE_S` |
| `pto.tfree_internal` | `pipe` | — | 否 | `PIPE_S` |

---

## 9. Tile 大小与 Slot Size

**slot_size 定义：** FIFO 中每个 slot 的大小，取生产者和消费者 tile 中较大的：

```
slot_size = max(srcTile.size, dstTile.size)
```

其中 `tile.size = Rows × Cols × sizeof(DType)`。

**srcTile 与 dstTile 的三种关系：**

| 关系 | 条件 | slot_size | 典型场景 |
|---|---|---|---|
| 1:1 相等 | `src.size == dst.size` | 任一 | A3（Cube/Vec 算力 1:1），A5 单 Vec 核 |
| Src 为 Dst 的 2 倍 | `src.size == 2 × dst.size` | src.size | A5 C2V（1 Cube : 2 Vector） |
| Dst 为 Src 的 2 倍 | `dst.size == 2 × src.size` | dst.size | A5 V2C |

**plan memory 分配量 = `local_fifo_depth × slot_size`**

---

## 10. C++ TPipe 模板参考

### 10.1 模板定义（pto-isa）

```cpp
template <uint8_t FlagID, FIFOType FiFoType, uint8_t FiFoDepth, uint8_t FiFoSyncT,
          typename TileDataProd, typename TileDataCons,
          bool EN_UNIT_FLAG = false,
          uint8_t LocalFiFoDepth = 2,
          VecCubeRatio VCRatio = VecCubeRatio::V2C1_VECS>
struct TPipe;
```

### 10.2 参数说明

| # | 参数 | 类型 | 说明 | PTOAS 来源 |
|---|---|---|---|---|
| 1 | `FlagID` | uint8_t | 同步 flag 基地址 | 编译器 flag 分配 pass |
| 2 | `FiFoType` | FIFOType | FIFO 类型 | 由 Op 类型 + dir_mask 推导 |
| 3 | `FiFoDepth` | uint8_t | ring buffer slot 数量 | dir_mask 推导（单向=8） |
| 4 | `FiFoSyncT` | uint8_t | 同步周期 | C++ 默认值（=FiFoDepth） |
| 5 | `TileDataProd` | typename | 生产者 tile 类型 | pipe result type 的 srcTileType |
| 6 | `TileDataCons` | typename | 消费者 tile 类型 | pipe result type 的 dstTileType |
| 7 | `EN_UNIT_FLAG` | bool | unit flag 优化 | C++ 默认值（false） |
| 8 | `LocalFiFoDepth` | uint8_t | GM_FIFO 的 local buffer 深度 | `local_fifo_depth` 属性，默认 2 |
| 9 | `VCRatio` | VecCubeRatio | Vector/Cube 核心比率 | C++ 默认值（V2C1_VECS） |

### 10.3 VecCubeRatio 枚举

```cpp
enum class VecCubeRatio : uint8_t {
    V1C1_VEC0 = 0,  // 1 Vector : 1 Cube，仅 Vector 0
    V1C1_VEC1 = 1,  // 1 Vector : 1 Cube，仅 Vector 1
    V2C1_VECS = 2,  // 2 Vector : 1 Cube（默认）
};
```

VCRatio 不可从 tile 大小推导——`src.size == dst.size` 时 V1C1 与 V2C1 均合法。当前使用 C++ 默认值 V2C1_VECS。

### 10.4 FIFOType 与行为

| FIFOType | 对应 Op | 构造函数参数 | TPUSH | TPOP |
|---|---|---|---|---|
| `VEC_FIFO` | `initialize_l2l_pipe` (C2V) | local addr | 写入 local VEC slot | 返回 local slot tile |
| `MAT_FIFO` | `initialize_l2l_pipe` (V2C) | local addr | 写入 local MAT slot | 返回 local slot tile |
| `GM_FIFO` | `initialize_l2g2l_pipe` | GM addr + local addr | 写入 GM slot | GM → local → 返回 tile |

---

## 11. 端到端示例：EmitC 输出

### 11.1 L2L Pipe（A5）

```cpp
__global__ AICORE void c2v_l2l(int32_t local_addr) {
  auto v28 = TPipe<0, FIFOType::VEC_FIFO, 8, 8, AccTile, VecTile>(local_addr);

  // section.cube
  TPUSH(acc_tile, v28);

  // section.vector
  Tile<TileType::Vec, float, 32, 128, ...> v30;
  TPOP(v30, v28);
  TMOV(dst, v30);
  TFREE(v28);
}
```

### 11.2 L2G2L Pipe（A3 或 A5）

```cpp
__global__ AICORE void c2v_l2g2l(memref gm_addr, int32_t local_addr) {
  auto v28 = TPipe<0, FIFOType::GM_FIFO, 8, 8, AccTile, VecTile>(gm_addr, local_addr);

  // section.cube
  TPUSH(acc_tile, v28);

  // section.vector
  Tile<TileType::Vec, float, 32, 128, ...> v30;
  TPOP(v30, v28);
  TMOV(dst, v30);
  TFREE(v28);
}
```
