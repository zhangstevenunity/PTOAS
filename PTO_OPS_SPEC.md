# PTO Dialect Op 规格文档

本文档基于 `include/PTO/IR/PTOOps.td`，按 PTO Dialect 的 op 定义整理出使用约定与 op 分类索引，便于在编写/解析 MLIR 时快速查阅。

同步相关说明：
- `pto.record_event` / `pto.wait_event` 的操作类型属性规范写法为 `#pto.pipe_event_type<...>`。
- `#pto.sync_op_type<...>` 为历史别名，仍可解析，但不建议继续使用。

## PTO Op IR 格式速查表（DPS 格式）

| 分类 | PTO | Summary | IR 格式 (DPS格式) |
|------|-----|---------|-------------------|
| View | make_tensor_view | Wrap a pointer as a tensor_view descriptor (no allocation, no copy). | `%dst = pto.make_tensor_view %ptr, shape = [sh1,sh2,sh3,sh4,sh5] strides = [st1,st2,st3,st4,st5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>` |
|  | partition_view | Partition a tensor view into a smaller logical view (logical slicing). | `%dst = pto.partition_view %src, offsets = [of1,of2,of3,of4,of5], sizes = [sh1,sh2,sh3,sh4,sh5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype> -> !pto.partition_tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>` |
|  | alloc_tile(静态参数) | Allocates a tile buffer (logical buffer). | `%dst = pto.alloc_tile : !pto.tile_buf<loc, dtype, rows, cols, v_row, v_col, blayout, slayou, fractal, pad>` |
|  | alloc_tile(动态参数) | Allocates a tile buffer (logical buffer). | `%dst = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc, dtype, rows, cols, v_row=?, v_col=?, blayout, slayou, fractal, pad>` |
| 获取核参数 | get_block_idx | Get block index. | `%idx  = pto.get_block_idx` |
|  | get_subblock_idx | Get sub-block index. | `%idx  = pto.get_subblock_idx` |
|  | get_block_num | Get block count. | `%num  = pto.get_block_num` |
|  | get_subblock_num | Get sub-block count. | `%num  = pto.get_subblock_num` |
|  | addptr |  | `%ptr_new = pto.addptr %ptr, %offset` |
|  | tgetval | Read a single element from tile-like src at offset into a scalar (tile world). | `pto.tgetval ins(%src, %index : !pto.tile_buf<...>, dtype) outs(%val : dtype)` |
|  | tsetval | Write a scalar value into a single element of dst at offset (tile world). | `pto.tsetval ins(%index, %val : dtype, dtype) outs(%dst : !pto.tile_buf<...>)` |
| 内存 | TLOAD | Load data from a GlobalTensor (GM) into a Tile. | `pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSTORE | Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters. | `pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)` |
|  | TSTORE_FP | Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters. `TSTORE_FP` is the fp-quantization overload of `TSTORE`. | `pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)` |
|  | MGATHER | Gather-load elements from global memory into a tile using per-element indices. | `pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | MSCATTER | Scatter-store elements from a tile into global memory using per-element indices. | `pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)` |
| 同步 | RECORD_EVENT | Record an event. | `pto.record_event[src_op, dst_op, eventID]` |
|  | WAIT_EVENT | Wait on an event. | `pto.wait_event[src_op, dst_op, eventID]` |
|  | BARRIER | Barrier op (global). | `pto.barrier(op)` |
| Tile x Tile | TADD | Elementwise add of two tiles. | `pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSUB | Elementwise subtract of two tiles. | `pto.tsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TMUL | Elementwise multiply of two tiles. | `pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TDIV | Elementwise division of two tiles. | `pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TREM | Elementwise remainder of two tiles. | `pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TAND | Elementwise bitwise AND of two tiles. | `pto.tand ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TOR | Elementwise bitwise OR of two tiles. | `pto.tor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TXOR | Elementwise bitwise XOR of two tiles. | `pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSHL | Elementwise shift-left of two tiles. | `pto.tshl ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSHR | Elementwise shift-right of two tiles. | `pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TMAX | Elementwise maximum of two tiles. | `pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TMIN | Elementwise minimum of two tiles. | `pto.tmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TPRELU | Elementwise PReLU (parametric ReLU) with a per-element slope tile. | `pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TCMP | Compare two tiles and write a packed predicate mask. | `pto.tcmp ins(%src0, %src1  {cmpMode = #pto<cmp xx>} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TABS | Elementwise absolute value of a tile. | `pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TEXP | Elementwise exponential. | `pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TLOG | Elementwise natural logarithm of a tile. | `pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSQRT | Elementwise square root. | `pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TRSQRT | Elementwise reciprocal square root. | `pto.trsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TRECIP | Elementwise reciprocal of a tile. | `pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TNEG | Elementwise negation of a tile. | `pto.tneg ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TNOT | Elementwise bitwise NOT of a tile. | `pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TRELU | Elementwise ReLU of a tile. | `pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TCVT | Elementwise type conversion with a specified rounding mode. | `pto.tcvt ins(%src {rmode = #pto<round_mode xx>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TADDC | Elementwise ternary add: `src0 + src1 + src2`. | `pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSUBC | Elementwise ternary op: `src0 - src1 + src2`. | `pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSEL | Select between two tiles using a mask tile (per-element selection). | `pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
| Tile × 标量 | TADDS | Elementwise add a scalar to a tile. | `pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TSUBS | Elementwise subtract a scalar from a tile. | `pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TMULS | Elementwise multiply a tile by a scalar. | `pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TDIVS | Elementwise division with a scalar (tile/scalar or scalar/tile). | `pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TREMS | Elementwise remainder with a scalar: `%`. | `pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TANDS | Elementwise bitwise AND of a tile and a scalar. | `pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TORS | Elementwise bitwise OR of a tile and a scalar. | `pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TXORS | Elementwise bitwise XOR of a tile and a scalar. | `pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TSHLS | Elementwise shift-left of a tile, shift bits given by scalar. | `pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TSHRS | Elementwise shift-right of a tile, shift bits given by scalar. | `pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TMAXS | Elementwise max of a tile and a scalar: `max(src, scalar)`. | `pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TMINS | Elementwise minimum of a tile and a scalar. | `pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TLRELU | Leaky ReLU with a scalar slope. | `pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TCMPS | Compare a tile against a scalar and write per-element comparison results. | `pto.tcmps ins(%src, %scalar {cmpMode = #pto<cmp xx>} : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TADDSC | Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`. | `pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSUBSC | Elementwise fused op: `src0 - scalar + src1`. | `pto.tsubsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSELS | Select one of two source tiles using a scalar `selectMode` (global select). For per-element selection, use `TSEL`. | `pto.tsels ins(%src0, %src1, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TEXPANDS | Broadcast a scalar into a destination tile. | `pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)` |
| 轴向归约/展开 | TROWSUM | Reduce each row by summing across columns. | `pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TROWMAX | Reduce each row by taking the maximum across columns. | `pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TROWMIN | Reduce each row by taking the minimum across columns. | `pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TROWEXPAND | Broadcast the first element of each source row across the destination row. | `pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TCOLSUM | Reduce each column by summing across rows. | `pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TCOLMAX | Reduce each column by taking the maximum across rows. | `pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TCOLMIN | Reduce each column by taking the minimum across rows. | `pto.tcolmin ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TCOLEXPAND | Broadcast the first element of each source column across the destination column. | `pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TROWEXPANDDIV | Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`. | `pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TROWEXPANDMUL | Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`. | `pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TROWEXPANDSUB | Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`. | `pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
| Padding | TFILLPAD | Copy a source tile into a destination tile and fill the remaining (padded) elements with a compile-time pad value selected by `TileDataDst::PadVal` (e.g., `PadValue::Min`/`PadValue::Max`). This is commonly used to materialize deterministic values outside the runtime valid region so that subsequent ops can operate on a full static tile shape. | `pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
| 矩阵乘 | TMATMUL | Matrix multiply (GEMM) producing an accumulator/output tile. | `pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)` |
|  | TMATMUL_MX | Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets. This instruction is currently implemented on A5. | `pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c :  !pto.tile_buf<...>)` |
|  | TMATMUL_ACC | Matrix multiply with accumulator input (fused accumulate). | `pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)` |
|  | TMATMUL_BIAS | Matrix multiply with bias add. | `pto.tmatmul.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)` |
|  | TGEMV | General Matrix-Vector multiplication (GEMV) producing an accumulator/output tile. | `pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)` |
| 数据搬运/布局 | TMOV | Move/copy between tiles, optionally applying implementation-defined conversion modes selected by template parameters and overloads. `TMOV` is used for: - Vec -> Vec moves - Mat -> Left/Right/Bias/Scaling/Scale(Microscaling) moves (target-dependent) - Acc -> Vec moves (target-dependent) | `pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TMOV_FP | Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters. `TMOV_FP` is a named wrapper around the `TMOV_IMPL(..., fp)` path and is part of the `TMOV` family. | `pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TTRANS | Transpose with an implementation-defined temporary tile. | `pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TEXTRACT | Extract a sub-tile from a source tile. | `pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TRESHAPE | Reinterpret a tile as another tile type/shape while preserving the underlying bytes. This is a *bitwise* reshape: it does not change values, it only changes how the same byte buffer is viewed. | `pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TASSIGN |  | `pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)` |
| 复杂指令 | TCI | Generate a contiguous integer sequence into a destination tile. | `pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)` |
|  | TGATHER | Gather/select elements using either an index tile or a compile-time mask pattern. | `pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TGATHERB | Gather elements using byte offsets. | `pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSCATTER | Scatter rows of a source tile into a destination tile using per-element row indices. | `pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TSORT32 |  | `pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)` |
|  | TMRGSORT | Merge sort for multiple sorted lists (implementation-defined element format and layout). | `pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)` |
|  | TPARTADD | Partial elementwise add with implementation-defined handling of mismatched valid regions. | `pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TPARTMAX | Partial elementwise max with implementation-defined handling of mismatched valid regions. | `pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
|  | TPARTMIN | Partial elementwise min with implementation-defined handling of mismatched valid regions. | `pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)` |
| 打印 | TPRINT | Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code. The TPRINT instruction outputs the logical view of data stored in a Tile or GlobalTensor. It supports common data types (e.g., float, half, int8, uint32) and multiple memory layouts (ND, DN, NZ for GlobalTensor; vector tiles for on-chip buffers). Important: This instruction is for development and debugging ONLY. It incurs significant runtime overhead and must not be used in production kernels. Output may be truncated if it exceeds the internal print buffer. Requires CCE compilation option -D_DEBUG --cce-enable-print | `pto.tprint ins(%src : !pto.tile_buf<...> \| !pto.partition_tensor_view<MxNxdtype>)` |
