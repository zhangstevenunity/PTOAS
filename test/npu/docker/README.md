First follow [../../../docker/README.md](../../../docker/README.md) to obtain the wheel.

```bash
# build
docker build . -t ptoas_cann:8.5.0

# use
docker run --rm -it --ipc=host --privileged \
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
pip install /mounted_home/pto*.whl
cd /mounted_home/work_code/ptoas_fork/test/npu/abs
bash ./compile.sh
```
