ARG PYTORCH_VERSION=1.12.0
ARG CUDA_VERSION=11.3
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-runtime
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*
ENV TZ=Asia/Tokyo 
RUN pip3 install -U pip wheel setuptools && \
    pip3 install -r requirements_docker.txt -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu`echo ${CUDA_VERSION} | sed -e 's/\.//'`.html&& \
    pip3 install -e .
ENTRYPOINT ["python3.9", "train.py"]
CMD ["base=train_default"]
