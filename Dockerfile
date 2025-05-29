FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-distutils \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

RUN python3.8 -m pip install --upgrade pip

RUN pip3 install opencv-python==4.8.0.74 numpy

RUN pip3 install pywaggle[all]==0.56.0

RUN pip3 install ultralytics

ENV DEBIAN_FRONTEND=dialog

WORKDIR /app

COPY . .

ENTRYPOINT ["python3.8", "main.py"]