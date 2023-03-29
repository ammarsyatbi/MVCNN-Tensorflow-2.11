FROM tensorflow/tensorflow:2.11.0


COPY ./ /opt/ml/code

WORKDIR /opt/ml/code

# Uncomment if needed to isntall CUDA
# RUN apt-key del 7fa2af80 && \ 
#     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub 

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get install libcupti-dev -y && \
    apt-get install openjdk-8-jre-headless -y && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r ./requirements.txt && \
    python -m pip install wheel && \
    python -m pip install sagemaker-training && \
    python -m pip install multi-model-server sagemaker-inference

ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/opt/ml/code"

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py