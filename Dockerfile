FROM nvidia/cuda:11.3.0-runtime-ubuntu18.04
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata
ENV TZ=Asia/Tokyo 
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip
RUN python3.9 -m pip install -U pip wheel setuptools && \
    python3.9 -m pip install torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 &&\
    python3.9 -m pip install -r requirements_docker.txt && \
    python3.9 -m pip install -e .
ENTRYPOINT ["python3.9", "train.py"]
CMD [base=train_default]
