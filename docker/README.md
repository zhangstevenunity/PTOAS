Build:

```bash
sudo docker build . -t ptoas:py3.11

# optional, to change python version
sudo docker build . -t ptoas:py3.12 --build-arg PY_VER=cp312-cp312
```

Use:

```bash
sudo docker run --rm -it \
    -v $HOME:/mounted_home -w /mounted_home \
    ptoas:py3.11 /bin/bash

which ptoas  # $PTO_SOURCE_DIR/build/tools/ptoas/ptoas

# copy wheel out of container to use in other environment
mkdir -p /mounted_home/pto_wheels
cp $PY_PACKAGE_DIR/wheelhouse/ptoas*.whl /mounted_home/pto_wheels/
cp $PTO_SOURCE_DIR/build/tools/ptoas/ptoas /mounted_home/pto_wheels/
```
