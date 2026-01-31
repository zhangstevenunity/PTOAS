FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11

RUN apt update && apt install -y zip

RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.8.0.post2

RUN pip install --no-cache-dir numpy

# TODO: install stable release from https://pypi.org/project/ptoas/#history
# Now install local wheel during docker runtime

WORKDIR /sources
RUN git clone https://gitcode.com/cann/pto-isa.git
# TODO: use a stable tag like:
# RUN git clone -b v8.5.0-beta.1 --depth 1 https://gitcode.com/cann/pto-isa.git
