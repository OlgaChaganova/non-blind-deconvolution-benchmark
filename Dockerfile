FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ENTRYPOINT ["/bin/bash", "-l", "-c"]

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub


RUN apt-get update -qq \
    && apt-get install build-essential -y \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update && apt-get -y install git \
    && apt-get install -y python3-pip

WORKDIR /nbdb

COPY . /nbdb/

RUN pip install -r requirements.txt

CMD ["python", "src", "test.py", "--table_name", "full_benchmark"]