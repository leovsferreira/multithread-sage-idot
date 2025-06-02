FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-distutils \
    wget \
    git \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

RUN python3.9 -m pip install --upgrade pip

RUN pip3 install "numpy<2" opencv-python==4.8.0.74
RUN pip3 install pywaggle[all]==0.56.0
RUN pip3 install ultralytics
RUN pip3 install pytz

WORKDIR /app

RUN mkdir -p /app/models

RUN python3.9 -c "from ultralytics import YOLO; import shutil; model = YOLO('yolov8n.pt'); shutil.move('yolov8n.pt', '/app/models/yolov8n.pt')"

RUN ls -la /app/models/yolov8n.pt && echo "Model downloaded successfully"

ENV DEBIAN_FRONTEND=dialog

COPY . .

ENTRYPOINT ["python3.9", "main.py"]