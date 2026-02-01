bisheng \
    -I${PTO_LIB_PATH}/include \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -I${ASCEND_TOOLKIT_HOME}/pkg_inc \
    -I${ASCEND_TOOLKIT_HOME}/profiling \
    -I${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime/runtime \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
    -xcce -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -std=gnu++17 \
    ./abs_vec_core.cpp \
    -o ./abs_kernel_full_flg.so
