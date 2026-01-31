First follow [../../../docker/README.md](../../../docker/README.md) to obtain the wheel.

```bash
# build
sudo docker build . -t ptoas_cann:8.5.0

# use
sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $HOME:/mounted_home \
    -w /mounted_home \
    ptoas_cann:8.5.0 \
    /bin/bash

# in container
pip install /mounted_home/pto_wheels/pto*.whl
cp /mounted_home/pto_wheels/ptoas /usr/local/bin/

cd /mounted_home/work_code/ptoas_fork/test/npu/abs

PY_PKG_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
# /usr/local/python3.11.14/lib/python3.11/site-packages for this particular image
export LD_LIBRARY_PATH=${PY_PKG_PATH}/ptoas.libs:$LD_LIBRARY_PATH

ldd /usr/local/bin/ptoas | grep libMLIRMlirOptMain  # still missing
bash ./compile.sh
```
