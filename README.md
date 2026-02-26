# PTOAS (PTO Assembler & Optimizer)

## 1. 项目简介 (Introduction)

**PTOAS** (`ptoas`) 是一个基于 **LLVM/MLIR (release/19.x)***(Commit e21dc4bd5474d04b8e62d7331362edcc5648d7e5)* 框架构建的专用编译器工具链，专为 **PTO Bytecode** (Programming Tiling Operator Bytecode) 设计。

项目仓库：[https://github.com/zhangstevenunity/PTOAS](https://github.com/zhangstevenunity/PTOAS)

作为连接上层 AI 框架与底层各类NPU/GPGPU/CPU硬件，`ptoas` 采用 **Out-of-Tree** 架构构建，提供了完整的 C++ 与 Python 接口，主要职责包括：

1. **IR 解析与验证**：解析 `.pto` 输入文件，验证 PTO Dialect 操作（Ops）的语义正确性。
2. **编译优化 (Passes)**：执行针对达芬奇架构（Da Vinci Architecture）的特定优化 Pass，如算子融合、自动同步插入策略等。
3. **代码生成 (Lowering)**：支持将 PTO IR 下降（Lowering）到 `EmitC` / `Linalg` Dialect，最终生成可调用 `pto-isa` C++ 库的代码。
4. **Python 绑定 (Python Bindings)**：提供无缝集成的 Python 模块。通过与 MLIR Core 绑定集成，支持 **PyPTO**、**TileLang**、**CuTile** 等框架在 Python 端直接构建、操作和编译 PTO Bytecode。

---

## 2. 目录结构 (Directory Structure)

```text
pto-project/
├── include/
│   └── PTO/               # PTO Dialect 的头文件与 TableGen 定义 (.td)
├── lib/
│   ├── PTO/               # Dialect 核心实现 (IR) 与 Pass 逻辑 (Transforms)
│   ├── CAPI/              # C 语言接口暴露
│   └── Bindings/Python/   # Python Binding C++ 实现 (Pybind11)
├── python/                # Python 模块构建脚本与辅助代码
├── test/
│   └── samples/            # 测试用例
├── tools/
│   └── pto-opt/           # ptoas 命令行工具入口 (Target: pto-opt, Output: ptoas)
└── CMakeLists.txt         # 顶级构建配置

```

---

## 3. 构建指南 (Build Instructions)

⚠️ **重要提示**：本项目严格依赖 **LLVM release/19.x** 分支。

### 3.0 环境变量配置 (Configuration)

为了简化构建流程，**请首先根据您的实际环境修改并运行以下命令**。后续步骤将直接引用这些变量。

```bash
# ================= 配置区域 (请修改这里) =================
# 设置您的工作根目录 (建议创建一个专门的目录存放 LLVM 和 PTOAS)
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM 源码与构建路径
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

# PTOAS 源码与安装路径
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
# =======================================================

# 创建工作目录
mkdir -p $WORKSPACE_DIR

```

### 3.1 环境准备 (Prerequisites)

* **OS**: Linux (Ubuntu 20.04+ 推荐)
* **Compiler**: GCC >= 9 或 Clang (支持 C++17)
* **Build System**: CMake >= 3.20, Ninja
* **Python**: 3.8+
* **Python Packages**: `pybind11`, `numpy`
```bash
pip3 install pybind11 numpy

```



### 3.2 第一步：构建 LLVM/MLIR (Dependency)

我们需要下载 LLVM 源码，切换到 `release/19.x` 分支，并以**动态库 (Shared Libs)** 模式编译，以确保 Python Binding 的正确链接。

```bash
# 1. 下载 LLVM 源码
cd $WORKSPACE_DIR
git clone https://github.com/llvm/llvm-project.git
cd $LLVM_SOURCE_DIR

# 2. [关键] 切换到 release/19.x 分支
git checkout release/19.x

# 3. 配置 CMake (构建动态库并启用 Python 绑定)
cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

# 4. 编译 LLVM (这一步耗时较长)
ninja -C $LLVM_BUILD_DIR

```

### 3.3 第二步：构建 PTOAS (Out-of-Tree)

下载 PTOAS 源码并基于刚刚编译好的 LLVM 19 进行构建。

```bash
# 1. 下载 PTOAS 源码
cd $WORKSPACE_DIR
git clone https://github.com/zhangstevenunity/PTOAS.git
cd $PTO_SOURCE_DIR

# 2. 获取 pybind11 的 CMake 路径
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

# 3. 配置 CMake
# 注意：此处直接使用了 3.0 章节中定义的变量，无需手动修改
# 目标硬件架构选择：-DPTOAS_TARGET_ARCH=A3 或 A5（默认 A5）
cmake -G Ninja \
    -S . \
    -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPTOAS_TARGET_ARCH=A5 \
    # 编译 Level 选择：level1 / level2 / level3（默认 level2；level3 会禁用 PlanMemory/InsertSync）
    -DPTOAS_BUILD_LEVEL=level2 \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython3_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
    -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

# 4. 编译并安装
ninja -C build
ninja -C build install

# 5. 检查构建产物
检查_pto*.so是不copy到了llvm的mlir_core
$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core/
└── mlir
    └── _mlir_libs
        └── _pto.cpython-311-*.so   ✅ 自动在这里

_pto.py还是在ptoas install 目录
./install/
└── mlir
    └── dialects
        ├── pto.py
        └── _pto_ops_gen.py

ptoas在build目录下
./build/
└── tools
    └── ptoas
        └── ptoas

```

---

## 4. 运行环境配置 (Runtime Environment)

构建完成后，需要配置环境变量以便系统能找到 Python 包和动态库。您可以将以下命令添加到 `.bashrc` 或启动脚本中。

```bash
# --- 运行时变量配置 (基于之前定义的路径) ---

# 1. Python Path: 拼接 MLIR Core 和 PTO Core
#    这样在 python 中 import mlir.dialects.pto 时能正确找到
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH

# 2. Library Path: 确保能加载 LLVM 和 PTO 的动态库 (.so)
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH

# 3. PATH: 将 ptoas 添加到命令行路径
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH

```

---

## 5. 使用方法 (Usage)

### 5.1 命令行工具 (CLI)

```bash
# 解析并打印 PTO IR
ptoas tests/input.pto

# 运行 AutoSyncInsert Pass
ptoas tests/input.pto --enable-insert-sync -o outputfile.cpp

# 指定目标硬件架构（A3 / A5）
ptoas tests/input.pto --pto-arch=a3 -o outputfile.cpp

# 指定构建 Level（level3 会禁用 PlanMemory/InsertSync）
ptoas tests/input.pto --pto-level=level3 -o outputfile.cpp

```

### 5.2 Python 接口 (Python API)

配置好环境变量后，PTO Dialect 将作为 `mlir.dialects` 的一部分被加载。

```python
from mlir.ir import Context, Module, Location
# [关键] 从 mlir.dialects 导入 pto，这是 Out-of-tree 绑定的标准用法
from mlir.dialects import pto

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx)
    module = Module.create()
    print("PTO Dialect registered successfully!")

```

### 5.3 运行测试

```bash
# 运行python binding 测试
cd ./test/samples/MatMul/
python3 ./tmatmulk.py > ./tmatmulk.pto

# 运行ptoas 测试
./build/tools/ptoas/ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
```

### 5.4 上板验证

该流程用于将 `test/samples` 下生成的 `.cpp`（PTOAS 输出）自动生成 NPU 验证用例，并在 NPU 上运行。

```bash
# 1) 生成 npu_validation 测试目录（会在当前 sample 目录下创建 npu_validation/）
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/Abs/abs-pto.cpp \
  --run-mode npu \
  --soc-version Ascend910B1

# 2) 运行验证（run.sh 无需额外参数）
test/samples/Abs/npu_validation/run.sh
```

说明：
- `npu_validation/` 下会生成 `abs_kernel.cpp / main.cpp / golden.py / compare.py / run.sh / CMakeLists.txt`
- `golden.py` 默认生成随机输入，输出默认全零（只保证输入/输出数量、shape、datatype 与 kernel 参数一致）
- `compare.py` 负责对比 `golden*.bin` 与 `output*.bin`，不一致时会报错

---
