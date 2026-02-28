# PTOAS (PTO Assembler & Optimizer)

## 1. 项目简介 (Introduction)

**PTOAS** (`ptoas`) 是一个基于 **LLVM/MLIR 19.1.7**（tag：`llvmorg-19.1.7`）构建的专用编译器工具链，专为 **PTO Bytecode** (Programming Tiling Operator Bytecode) 设计。

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

⚠️ **重要提示**：本项目严格依赖 **LLVM/MLIR 19.1.7**。推荐使用 `llvmorg-19.1.7` 的 tag，以保证复现性。

### 3.0 环境变量配置 (Configuration)

为了简化构建流程，**请首先根据您的实际环境修改并运行以下命令**。后续步骤将直接引用这些变量。

```bash
# ================= 配置区域 (请修改这里) =================
# 设置您的工作根目录 (建议创建一个专门的目录存放 LLVM 和 PTOAS)
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM 源码与构建路径
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-llvm
# LLVM/MLIR 的 CMake package 路径（默认按源码构建方式；如果你用预编译包，会在 3.2A 覆盖它）
export LLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm
export MLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir

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

PTOAS 依赖 **LLVM/MLIR 19.1.7**。推荐使用 `llvmorg-19.1.7`。

说明：
- **官方预编译 LLVM 安装包**通常是 **static**（可用 `llvm-config --shared-mode` 确认；如果输出 `static`，等价于 `-DBUILD_SHARED_LIBS=OFF`）。
- 如果你需要运行 `test/samples` 下的 Python 例子，需要 **MLIR Python bindings**，这通常意味着你需要从源码构建 LLVM 并开启 `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`。

#### 方式 A：使用官方/预编译 LLVM（推荐，BUILD_SHARED_LIBS=OFF）

以 macOS Apple Silicon（arm64）为例：

```bash
export LLVM_VER=19.1.7

# 若配置时报错找不到 zstd 的 CMake target（例如 `zstd::libzstd_static`），请先安装 zstd
# macOS 推荐用 brew 装：
brew install zstd

cd $WORKSPACE_DIR
curl -L -o LLVM-${LLVM_VER}-macOS-ARM64.tar.xz \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/LLVM-${LLVM_VER}-macOS-ARM64.tar.xz
tar -xJf LLVM-${LLVM_VER}-macOS-ARM64.tar.xz

export LLVM_INSTALL_DIR=$WORKSPACE_DIR/LLVM-${LLVM_VER}-macOS-ARM64
export LLVM_DIR=$LLVM_INSTALL_DIR/lib/cmake/llvm
export MLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir
```

其他平台请在对应 tag 的 Release 页面选择合适的 `LLVM-<ver>-<OS>-<ARCH>.tar.xz` 包。

#### 方式 B：从源码构建 LLVM/MLIR（需要 MLIR Python 时使用）

```bash
# 1. 选择 LLVM 版本（tag）
export LLVM_TAG=llvmorg-19.1.7

# 2. 下载 LLVM 源码
cd $WORKSPACE_DIR
git clone --depth 1 --branch ${LLVM_TAG} https://github.com/llvm/llvm-project.git
cd $LLVM_SOURCE_DIR

# 3. 配置 CMake（static 库，开启 MLIR Python bindings）
cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

# 4. 编译 LLVM (这一步耗时较长)
ninja -C $LLVM_BUILD_DIR

```

### 3.3 第二步：构建 PTOAS (Out-of-Tree)

下载 PTOAS 源码并基于 LLVM/MLIR 19 进行构建。

```bash
# 1. 下载 PTOAS 源码
cd $WORKSPACE_DIR
git clone https://github.com/zhangstevenunity/PTOAS.git
cd $PTO_SOURCE_DIR

# 2. 配置 CMake
# - 如果你使用的是官方预编译 LLVM（方式 A），建议先关掉 Python binding（避免依赖 MLIR Python 包目录）
# - 如果你需要 Python binding（方式 B），保持 PTO_ENABLE_PYTHON_BINDING=ON，并传入 MLIR_PYTHON_PACKAGE_DIR
cmake -G Ninja \
    -S . \
    -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DPTO_ENABLE_PYTHON_BINDING=OFF \
    -DLLVM_DIR=$LLVM_DIR \
    -DMLIR_DIR=$MLIR_DIR \
    -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

# 3. 编译（CLI 工具）
ninja -C build pto-opt

# 4. 检查构建产物
./build/tools/ptoas/ptoas --version

# （可选）如果要启用 Python binding：需要 MLIR Python bindings（方式 B）并重新配置
# export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
# cmake -G Ninja -S . -B build \
#   -DPTO_ENABLE_PYTHON_BINDING=ON \
#   -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
#   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
#   -DPython3_EXECUTABLE=$(which python3) \
#   -DPython3_FIND_STRATEGY=LOCATION \
#   -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
#   -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
#   -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"
# ninja -C build && ninja -C build install

```

---

## 4. 运行环境配置 (Runtime Environment)

如果你启用了 Python binding（方式 B），需要配置环境变量以便系统能找到 Python 包和动态库。您可以将以下命令添加到 `.bashrc` 或启动脚本中。

```bash
# --- 运行时变量配置 (基于之前定义的路径) ---

# 仅当你从源码构建 LLVM 并启用了 MLIR Python bindings（方式 B）时，才需要设置 PYTHONPATH：
# export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
# export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
# export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH

# Library Path：当 LLVM 是 shared 模式时可能需要；官方预编译 LLVM 通常为 static，可按需设置
# export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH

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

## 6. 本地复刻 CI 验证 (Local CI Repro)

当 GitHub Actions 暂不可用时，建议在合入前至少在本地跑通一遍 CI 的关键流程。下面命令尽量与仓库里的 workflow 保持一致（见 `.github/workflows/ci.yml` / `.github/workflows/build_wheel.yml`）。

### 6.1 CI Job：`build-cli-prebuilt`（Ubuntu 22.04，预编译 LLVM，关闭 Python binding）

该流程验证：
- 预编译 LLVM/MLIR（static）可用
- `ptoas` 可成功构建
- 基础 smoke：`test/samples/InjectSync/*.pto` 能够转换为 `.cpp`

> 注意：官方预编译 LLVM 的 `libLLVM*.a` 里可能包含 LLVM bitcode（ThinLTO），因此建议用预编译包自带的 `clang++` + `lld` 来链接（否则可能遇到 `file format not recognized`）。

```bash
set -euo pipefail

# 0) 依赖
sudo apt-get update
sudo apt-get install -y \
  cmake git ninja-build clang lld \
  libedit-dev zlib1g-dev libxml2-dev libzstd-dev \
  xz-utils curl

# 1) 下载官方预编译 LLVM（Linux x86_64）
export LLVM_VER=19.1.7
mkdir -p llvm-prebuilt
cd llvm-prebuilt
curl -L -o "LLVM-${LLVM_VER}-Linux-X64.tar.xz" \
  "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VER}/LLVM-${LLVM_VER}-Linux-X64.tar.xz"
tar -xJf "LLVM-${LLVM_VER}-Linux-X64.tar.xz"
export LLVM_INSTALL_DIR="$PWD/LLVM-${LLVM_VER}-Linux-X64"
cd ..

# 2) 构建 PTOAS（关闭 Python binding）
export PATH="${LLVM_INSTALL_DIR}/bin:${PATH}"
cmake -G Ninja -S . -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DPTO_ENABLE_PYTHON_BINDING=OFF \
  -DLLVM_DIR="${LLVM_INSTALL_DIR}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_INSTALL_DIR}/lib/cmake/mlir" \
  -DCMAKE_C_COMPILER="${LLVM_INSTALL_DIR}/bin/clang" \
  -DCMAKE_CXX_COMPILER="${LLVM_INSTALL_DIR}/bin/clang++" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build pto-opt
./build/tools/ptoas/ptoas --version

# 3) Smoke：把 InjectSync 下的 .pto 转成 .cpp
out="$(mktemp -d)"
for f in test/samples/InjectSync/*.pto; do
  base="$(basename "$f" .pto)"
  ./build/tools/ptoas/ptoas "$f" --enable-insert-sync -o "${out}/${base}.cpp"
done
echo "Smoke outputs: ${out}"
```

### 6.2 Python binding 验证（对应 CI 的 `Build Wheel` 核心检查）

官方预编译 LLVM **不包含** MLIR Python 包目录（`mlir_core`），因此要验证 Python binding，需要从源码构建 LLVM/MLIR 并开启 `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`（建议 shared 模式，和 wheel CI 一致）。

```bash
set -euo pipefail

# 0) 假设你已按 3.0 配好 WORKSPACE_DIR / LLVM_SOURCE_DIR / LLVM_BUILD_DIR / PTO_SOURCE_DIR / PTO_INSTALL_DIR
export LLVM_TAG=llvmorg-19.1.7

# 1) 构建 LLVM/MLIR（shared + MLIR Python bindings）
cd "$WORKSPACE_DIR"
git clone --depth 1 --branch "${LLVM_TAG}" https://github.com/llvm/llvm-project.git
cd "$LLVM_SOURCE_DIR"
cmake -G Ninja -S llvm -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python3)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host"
ninja -C "$LLVM_BUILD_DIR"

# 2) 构建 PTOAS（开启 Python binding）
cd "$PTO_SOURCE_DIR"
export PYBIND11_CMAKE_DIR="$(python3 -m pybind11 --cmakedir)"
cmake -G Ninja -S . -B build \
  -DPTO_ENABLE_PYTHON_BINDING=ON \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DPython3_EXECUTABLE="$(which python3)" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
  -DMLIR_PYTHON_PACKAGE_DIR="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core" \
  -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"
ninja -C build
ninja -C build install

# 3) Python import 检查（CI 的 wheel import 检查等价）
export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$PTO_INSTALL_DIR:$PYTHONPATH"
python3 -c "import mlir.ir; print('mlir.ir OK')"
python3 -c "from mlir.dialects import pto; print('mlir.dialects.pto OK')"

# 4) （可选，推荐）跑 runop 覆盖更多 sample（只做 .py -> .pto -> .cpp 生成，不上板）
PTOAS_BIN="$PTO_SOURCE_DIR/build/tools/ptoas/ptoas" \
PYTHON_BIN="$(which python3)" \
bash test/samples/runop.sh -t MatMul

# 全量跑（耗时更久）
PTOAS_BIN="$PTO_SOURCE_DIR/build/tools/ptoas/ptoas" \
PYTHON_BIN="$(which python3)" \
bash test/samples/runop.sh all
```

### 6.3 可选：复刻 `remote-npu-validation`（需要可访问的远程 NPU 机器）

CI 上板验证依赖远程 NPU 环境与 `pto-isa`。你也可以本地生成 payload 后手动跑（命令参数见 `test/npu_validation/scripts/run_remote_npu_validation.sh`）。
