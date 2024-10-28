FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL maintainer="rnhhy"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip 

COPY requirements.txt /opt/python3/venv/base/
RUN python3 -m pip install --no-cache-dir -r /opt/python3/venv/base/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]