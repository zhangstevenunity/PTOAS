# PTOAS 新增 OP 接口与 TODO

## 1. 范围与目标

说明在PTOAS中新增的 OP 接口定义与参数语义，包括：

- `pto.initialize_pipe`
- `pto.tpush(tile, fifo)`
- `pto.tpop(tile, fifo)`

并补充当前待完成TODO项。

## 2. OP 接口定义

### 2.1 `pto.initialize_pipe`

用途：在函数级完成 ring buffer/FIFO 句柄初始化，返回统一 `fifo` 句柄，供后续 `tpush/tpop` 显式传递。

概念签名：

```mlir
%fifo = pto.initialize_pipe {dir_mask = <i8>, slot_size = <i32>}
  (%gm_slot_buffer : <PTODpsType>,
   %c2v_consumer_buf : i32,
   %v2c_consumer_buf : i32)
  -> !pto.fifo<SrcTileTy, DstTileTy, Loc, Depth, NumBuffers>
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `dir_mask` | `i8 attr` | 方向掩码，`1`=C2V，`2`=V2C，`3`=双向 | 当前 PTOAS 中 `3` 暂不支持，直接报错 |
| `slot_size` | `i32 attr` | 单 slot 大小（字节） | 必须 `> 0` |
| `gm_slot_buffer` | `PTODpsType` | GM slot buffer 基址/句柄 | 必填 |
| `c2v_consumer_buf` | `i32` | A5 下 C2V consumer 侧本地 buffer 基址 | 必填 |
| `v2c_consumer_buf` | `i32` | A5 下 V2C consumer 侧本地 buffer 基址 | 必填 |
| 返回值 `fifo` | `!pto.fifo<...>` | 统一 FIFO 句柄 | 必须为合法 `!pto.fifo` 类型 |

### 2.2 `pto.tpush(tile, fifo)`

用途：生产者将 tile 按 `fifo` 描述写入 ring buffer。

概念签名：

```mlir
pto.tpush(%tile, %fifo : <TileTy>, !pto.fifo<SrcTileTy, DstTileTy, Loc, Depth, NumBuffers>)
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `tile` | `PTODpsType` | 生产者要推送的 tile 变量 | `tile` 类型必须匹配 `fifo.src_tile_type` |
| `fifo` | `!pto.fifo<...>` | `initialize_pipe` 返回的 fifo 句柄 | 必须显式传入，不能隐式推导 |

### 2.3 `pto.tpop(tile, fifo)`

用途：消费者从 `fifo` 读取 tile 数据到目标 `tile` 变量。

概念签名：

```mlir
pto.tpop(%tile, %fifo : <TileTy>, !pto.fifo<SrcTileTy, DstTileTy, Loc, Depth, NumBuffers>)
```

参数说明：

| 参数 | 类型 | 说明 | 约束 |
|---|---|---|---|
| `tile` | `PTODpsType` | 消费者接收数据的 tile 变量 | `tile` 类型必须匹配 `fifo.dst_tile_type` |
| `fifo` | `!pto.fifo<...>` | `initialize_pipe` 返回的 fifo 句柄 | 必须显式传入，不能隐式推导 |

语义备注：

- 按目标方案，`tpop` 应只负责“获取 slot + 读取数据”。
- slot 释放应由独立 `tfree` 完成（当前 PTOAS 待补齐）。

## 3. TODO（当前版本）

### T1. `TFREE` 指令链路补齐

- 新增 `pto.tfree(fifo)`（ODS、verifier、EmitC lowering、测试）。
- 明确 `tpop/tfree` 配对规则，至少覆盖线性路径校验。
- 验收：`emit-c` 中释放时机由 `TFREE` 控制，不再隐式绑定到 `TPOP`。

### T2. `reserve_buffer/import_peer_buffer` 与常量传播

- 补齐 PTOAS IR 节点：`reserve_buffer`、`import_peer_buffer`。
- 在地址分配链路中支持 `base=auto`、冲突检查、peer 常量解析。
- 增加跨函数常量传播：consumer 的解析 base 传递到 producer 的 `import_peer_buffer`。

### T3. FlagID 分配策略重构

- 当前（`0,2,4,6,8,10,12,14`）的线性分配的策略较简单。
- 应该在kernel函数范围内进行分析和分配。
