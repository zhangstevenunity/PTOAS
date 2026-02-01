bisheng \
    -I${PTO_LIB_PATH}/include \
    -fPIC -shared -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    ./abs_vec_core.cpp \
    -o ./abs_kernel_min_flg.so

# NOTE: with CANN installation, even the `-I${PTO_LIB_PATH}/include` can be omitted
# because `$ASCEND_TOOLKIT_HOME/include/pto` also contains PTO headers (not most recent version) 