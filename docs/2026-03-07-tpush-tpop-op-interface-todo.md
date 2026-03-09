# PTOAS 新增 OP 接口与 TODO

## 1. 范围与目标

说明在PTOAS中新增的 OP 接口定义与参数语义，包括：

- `pto.initialize_pipe`
- `pto.tpush(tile, pipe)`
- `pto.tpop(tile, pipe)`
- `pto.tfree(pipe)`

并补充当前待完成TODO项。

## 2. OP 接口定义

### 2.1 `pto.initialize_pipe`

用途：在函数级完成 ring buffer/pipe 句柄初始化，返回统一 `pipe` 句柄，供后续 `tpush/tpop` 显式传递。

概念签名：

```mlir
%pipe = pto.initialize_pipe {dir_mask = <i8>, slot_size = <i32>}
  (%gm_slot_buffer : <PTODpsType>,
   %c2v_consumer_buf : i32,
   %v2c_consumer_buf : i32)
  -> !pto.pipe<SrcTileTy, DstTileTy>
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `dir_mask` | `i8 attr` | 方向掩码，`1`=C2V，`2`=V2C，`3`=双向 | 当前 PTOAS 中 `3` 暂不支持，直接报错 |
| `slot_size` | `i32 attr` | 单 slot 大小（字节） | 必须 `> 0` |
| `gm_slot_buffer` | `PTODpsType` | GM slot buffer 基址/句柄 | 必填 |
| `c2v_consumer_buf` | `i32` | A5 下 C2V consumer 侧本地 buffer 基址 | 必填 |
| `v2c_consumer_buf` | `i32` | A5 下 V2C consumer 侧本地 buffer 基址 | 必填 |
| 返回值 `pipe` | `!pto.pipe<SrcTileTy, DstTileTy>` | 统一 pipe 句柄 | `location/depth/numBuffers` 由 `initialize_pipe` 参数推导 |

### 2.2 `pto.tpush(tile, pipe)`

用途：生产者将 tile 按 `pipe` 描述写入 ring buffer。

概念签名：

```mlir
pto.tpush(%tile, %pipe : <TileTy>, !pto.pipe<SrcTileTy, DstTileTy>)
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `tile` | `PTODpsType` | 生产者要推送的 tile 变量 | `tile` 类型必须匹配 `pipe.src_tile_type` |
| `pipe` | `!pto.pipe<SrcTileTy, DstTileTy>` | `initialize_pipe` 返回的 pipe 句柄 | 必须显式传入，不能隐式推导 |

### 2.3 `pto.tpop(tile, pipe)`

用途：消费者从 `pipe` 读取 tile 数据到目标 `tile` 变量。

概念签名：

```mlir
pto.tpop(%tile, %pipe : <TileTy>, !pto.pipe<SrcTileTy, DstTileTy>)
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `tile` | `PTODpsType` | 消费者接收数据的 tile 变量 | `tile` 类型必须匹配 `pipe.dst_tile_type` |
| `pipe` | `!pto.pipe<SrcTileTy, DstTileTy>` | `initialize_pipe` 返回的 pipe 句柄 | 必须显式传入，不能隐式推导 |

语义备注：

- `tpop` 只负责”获取 slot + 读取数据”。
- slot 释放由独立的 `pto.tfree` 完成（仅 A5 架构需要，见 2.4）。

### 2.4 `pto.tfree(pipe)`

用途：显式释放 `tpop` 占用的 pipe slot。仅 A5 架构需要——A5 使用 Local buffer 作为 push/pop 数据传递介质，`tpop` 后数据仍在 slot 中供后续计算读取，必须等消费者用完后才能释放。A2A3 使用 Global Memory 通信，`tpop` 已将数据拷贝至本地内存，slot 可立即释放，因此 `tfree` 在 A2A3 上为空操作（EmitC 直接擦除）。

概念签名：

```mlir
pto.tfree(%pipe : !pto.pipe<SrcTileTy, DstTileTy>)
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `pipe` | `!pto.pipe<SrcTileTy, DstTileTy>` | `initialize_pipe` 返回的 pipe 句柄 | 必须与对应 `tpop` 使用同一 pipe |

约束与行为：

- 必须在 `section.cube` 或 `section.vector` 内部使用。
- 每个 `tpop` 应对应一个 `tfree`，使用相同的 `pipe_handle`。
- `InsertTFreePass`（仅 A5）会在 `tpop` 的 tile 数据最后一次被读取之后自动插入 `tfree`；已有手写 `tfree` 的 `tpop` 会被跳过。
- EmitC 降低：A5 生成 `TFREE(...)`，A2A3 擦除该 op。

## 3. TODO（当前版本）

### T1. FlagID 分配策略重构

- 当前（`0,2,4,6,8,10,12`）的线性分配策略较简单。
- 应该在 kernel 函数范围内进行分析和分配。
