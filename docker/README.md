Build:

```bash
docker build . -t ptoas:py3.12

# optional, to change python version
docker build . -t ptoas:py3.11 --build-arg PY_VER=cp311-cp311
```

Use:

```bash
docker run --rm -it \
    -v $HOME:/mounted_home -w /mounted_home \
    ptoas:py3.12 /bin/bash

which ptoas  # /opt/python/cp312-cp312/bin/ptoas
```
