python ./relu_builder.py > relu.pto
ptoas relu.pto -o relu_generated.cpp 2>debug_as.log

# manually change: relu_generated.cpp -> relu_edited.cpp 
# TODO: use a Python script to patch source code

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${PTO_LIB_PATH}/include \
    ./relu_edited.cpp \
    -o ./relu_kernel.so
