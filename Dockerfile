ARG PYTORCH_VERSION=1.12.0
ARG CUDA_VERSION=11.3.0
ARG CUDA_VERSION_SHORT=113

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu18.04
ARG PYTORCH_VERSION=1.12.0
ARG CUDA_VERSION=11.3.0
ARG CUDA_VERSION_SHORT=113
WORKDIR /app
COPY . .
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    software-properties-common \
    tzdata \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*
ENV TZ=Asia/Tokyo
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get -y install --no-install-recommends\
    python3.9 \
    python3.9-distutils \
    python3-pip
RUN python3.9 -m pip install -U pip wheel setuptools && \
    python3.9 -m pip install torch==${PYTORCH_VERSION}+cu${CUDA_VERSION_SHORT} -f https://download.pytorch.org/whl/torch_stable.html && \
    python3.9 -m pip install -r requirements_docker.txt -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu${CUDA_VERSION_SHORT}.html&& \
    python3.9 -m pip install .
ENTRYPOINT ["python3.9", "src/pyggnns/cli/train.py"]
CMD ["experiment=default"]
