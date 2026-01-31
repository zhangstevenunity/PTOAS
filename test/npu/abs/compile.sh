python ./abs.py > ./abs.pto
ptoas ./abs.pto --enable-insert-sync -o ./abs.cpp

# TODO: automate abs.cpp -> abs_modified.cpp conversion

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    -DMEMORY_BASE \
    --npu-arch=dav-2201 \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -I${ASCEND_TOOLKIT_HOME}/tools \
    -I${PTO_LIB_PATH}/include/pto \
    -o abs_lib.so abs_modified.cpp

# TODO: fix bisheng command according to https://gitcode.com/cann/pto-isa/pull/221
# or https://github.com/tile-ai/tilelang-ascend/blob/d2405b5975edaa36166b702d2ef8c1952b7b7337/tilelang/jit/adapter/libgen.py#L71
