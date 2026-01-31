ARG ARCH=x86_64
# ARCH options: x86_64, aarch64

# -----------------------------------------------------------------------------
# Stage 1: build wheel and ptoas in manylinux image
# -----------------------------------------------------------------------------
FROM quay.io/pypa/manylinux_2_34_${ARCH} AS builder
ARG ARCH
# need to re-declare ARG after FROM

# NOTE: change $PY_VER for different Python versions (3.8 - 3.14 available)
ARG PY_VER=cp311-cp311
ARG LLVM_TAG=llvmorg-19.1.7

## -- usually no need to change below --

ENV PY_PATH="/opt/python/${PY_VER}"
ENV PATH="${PY_PATH}/bin:${PATH}"

# dependency
RUN dnf install -y ninja-build cmake git ccache gcc-c++ lld zip && dnf clean all
RUN pip install --no-cache-dir numpy pybind11 nanobind setuptools wheel auditwheel

# setting build directories
ENV WORKSPACE_DIR=/llvm-workspace
ENV LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
ENV LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

ENV PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
ENV PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

# build LLVM
WORKDIR $WORKSPACE_DIR
RUN git clone --depth 1 --branch ${LLVM_TAG} https://github.com/llvm/llvm-project.git

WORKDIR $LLVM_SOURCE_DIR
RUN cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=${PY_PATH}/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

RUN ninja -C $LLVM_BUILD_DIR

# build ptoas
WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/zhangstevenunity/PTOAS.git
# TODO: tag git commit of PTOAS repo

WORKDIR $PTO_SOURCE_DIR
RUN cmake -G Ninja \
    -S . \
    -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPython3_ROOT_DIR=${PY_PATH} \
    -DPython3_EXECUTABLE=${PY_PATH}/bin/python \
    -DPython3_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
    -DMLIR_PYTHON_PACKAGE_DIR=${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core \
    -DCMAKE_INSTALL_PREFIX=${PTO_INSTALL_DIR}

RUN ninja -C build && ninja -C build install

# create python wheel
ENV PY_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core/

# copy pto.py, _pto_ops_gen.py 
RUN cp $PTO_INSTALL_DIR/mlir/dialects/*.py $PY_PACKAGE_DIR/mlir/dialects/

COPY ./setup.py $PY_PACKAGE_DIR/

WORKDIR $PY_PACKAGE_DIR
RUN python setup.py bdist_wheel

# fix missing so
RUN export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH \
    && auditwheel repair --plat manylinux_2_34_${ARCH} dist/ptoas*.whl

RUN pip install wheelhouse/ptoas*.whl

# test import without adjusting $PYTHONPATH and $LD_LIBRARY_PATH, in some irrelevant dir
WORKDIR /test
RUN python -c "import mlir.ir"
RUN python -c "from mlir.dialects import pto"

# test ptoas cli
ENV PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
RUN which ptoas

WORKDIR $PTO_SOURCE_DIR/test/samples/MatMul/
RUN python ./tmatmulk.py > ./tmatmulk.pto
RUN ptoas ./tmatmulk.pto -o ./tmatmulk.cpp

WORKDIR $PTO_SOURCE_DIR/test/samples/Abs/
RUN python ./abs.py > ./abs.pto
RUN ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp

# NOTE: compilation of the cpp file using `bisheng` needs full CANN image (10 GB)

# -----------------------------------------------------------------------------
# Stage 2: verify wheel and ptoas in ascend/python image (no build tree)
# -----------------------------------------------------------------------------
FROM quay.io/ascend/python:3.11-ubuntu22.04 AS runtime

# Copy only the wheel, ptoas binary, its .so dependencies, and test scripts from builder
COPY --from=builder /llvm-workspace/llvm-project/build-shared/tools/mlir/python_packages/mlir_core/wheelhouse/ptoas*.whl /tmp/
COPY --from=builder /llvm-workspace/PTOAS/build/tools/ptoas/ptoas /usr/local/bin/ptoas

RUN mkdir -p /usr/local/lib/ptoas
COPY --from=builder /llvm-workspace/llvm-project/build-shared/lib/. /usr/local/lib/ptoas/
COPY --from=builder /llvm-workspace/PTOAS/install/lib/. /usr/local/lib/ptoas/
ENV LD_LIBRARY_PATH=/usr/local/lib/ptoas:${LD_LIBRARY_PATH}

COPY --from=builder /llvm-workspace/PTOAS/test/samples/MatMul/tmatmulk.py /test/MatMul/
COPY --from=builder /llvm-workspace/PTOAS/test/samples/Abs/abs.py /test/Abs/

RUN pip install --no-cache-dir /tmp/ptoas*.whl && rm /tmp/ptoas*.whl

# test import without adjusting $PYTHONPATH and $LD_LIBRARY_PATH, in some irrelevant dir
WORKDIR /test
RUN python -c "import mlir.ir"
RUN python -c "from mlir.dialects import pto"

# test ptoas cli
RUN which ptoas

# same MatMul/Abs sample tests as in builder
WORKDIR /test/MatMul
RUN python ./tmatmulk.py > ./tmatmulk.pto
RUN ptoas ./tmatmulk.pto -o ./tmatmulk.cpp

WORKDIR /test/Abs
RUN python ./abs.py > ./abs.pto
RUN ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp

WORKDIR /test

CMD ["/bin/bash"]
