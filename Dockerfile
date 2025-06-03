FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

RUN python3.11 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    "numpy<2" \
    opencv-python==4.8.0.74 \
    pywaggle[all]==0.56.0 \
    ultralytics \
    pytz

WORKDIR /app

RUN mkdir -p /app/models

RUN python3.11 -c "from ultralytics import YOLO; import shutil; model = YOLO('yolov8n.pt'); shutil.move('yolov8n.pt', '/app/models/yolov8n.pt'); model = YOLO('yolov5nu.pt'); shutil.move('yolov5nu.pt', '/app/models/yolov5nu.pt'); model = YOLO('yolov10n.pt'); shutil.move('yolov10n.pt', '/app/models/yolov10n.pt')"

RUN ls -la /app/models/ && echo "All models downloaded successfully"

ENV DEBIAN_FRONTEND=dialog

COPY . .

ENTRYPOINT ["python3.11", "main.py"]