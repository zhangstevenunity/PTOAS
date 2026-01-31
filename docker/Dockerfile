ARG ARCH=x86_64
FROM quay.io/pypa/manylinux_2_34_${ARCH}

# NOTE: change $PY_VER for different Python versions (3.8 - 3.14 available)
ARG PY_VER=cp311-cp311
ARG LLVM_TAG=llvmorg-19.1.7

## -- usually no need to change below --

ENV PY_PATH="/opt/python/${PY_VER}"
ENV PATH="${PY_PATH}/bin:${PATH}"

# dependency
RUN dnf install -y ninja-build cmake git ccache gcc-c++ lld && dnf clean all
RUN pip install --no-cache-dir numpy pybind11 nanobind

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
RUN git clone -b npu https://github.com/learning-chip/PTOAS.git
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

# test ptoas cli exists (using full path, not PATH modification)
RUN test -f $PTO_SOURCE_DIR/build/tools/ptoas/ptoas && echo "ptoas binary found"

# create python wheel
ENV PY_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core/

# copy pto.py, _pto_ops_gen.py 
RUN cp $PTO_INSTALL_DIR/mlir/dialects/*.py $PY_PACKAGE_DIR/mlir/dialects/

COPY ./setup.py $PY_PACKAGE_DIR/

WORKDIR $PY_PACKAGE_DIR
RUN pip install --no-cache-dir setuptools wheel auditwheel
RUN python setup.py bdist_wheel

# fix missing so
ARG ARCH
RUN export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH \
    && auditwheel repair dist/ptoas*.whl

# RUN dnf install -y zip && dnf clean all

# # Insert ptoas binary into wheel; pip installs {name}-{version}.data/scripts/ into $PY_PATH/bin
# RUN set -e; \
#     WHEEL=$(ls wheelhouse/ptoas*.whl); \
#     EXTRACT=wheel_extract && mkdir -p "$EXTRACT" && unzip -q -o "$WHEEL" -d "$EXTRACT"; \
#     DATA_BASE=$(basename "$EXTRACT"/ptoas*.dist-info .dist-info); \
#     DATA_DIR="$EXTRACT/$DATA_BASE.data"; \
#     mkdir -p "$DATA_DIR/scripts"; \
#     cp "$PTO_SOURCE_DIR/build/tools/ptoas/ptoas" "$DATA_DIR/scripts/ptoas" && chmod +x "$DATA_DIR/scripts/ptoas"; \
#     RECORD=$(ls "$EXTRACT"/ptoas*.dist-info/RECORD); \
#     python3 -c "import hashlib, base64, os; p='$DATA_DIR/scripts/ptoas'; f=open(p,'rb'); h=base64.urlsafe_b64encode(hashlib.sha256(f.read()).digest()).decode().rstrip('='); f.close(); s=os.path.getsize(p); open('$RECORD','a').write(f'$DATA_BASE.data/scripts/ptoas,sha256={h},{s}\n')"; \
#     rm -f "$WHEEL"; \
#     (cd "$EXTRACT" && zip -q -r "../wheelhouse/$(basename $WHEEL)" .); \
#     rm -rf "$EXTRACT"

RUN pip install wheelhouse/ptoas*.whl

# test import without adjusting $PYTHONPATH and $LD_LIBRARY_PATH, in some irrelevant dir
WORKDIR /test
RUN python -c "from mlir.ir import Context, Location, Module, InsertionPoint"
RUN python -c "from mlir.dialects import pto"

WORKDIR $PTO_SOURCE_DIR/test/samples/MatMul/
RUN python ./tmatmulk.py > ./tmatmulk.pto

ENV PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
RUN ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
# TODO: test compilation of the cpp file using `bisheng`
# TODO: also test `ptoas --enable-insert-sync`

# show the built wheel by default
WORKDIR $PY_PACKAGE_DIR/wheelhouse

CMD ["/bin/bash"]
