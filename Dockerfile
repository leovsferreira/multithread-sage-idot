FROM python:3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.cache/torch/hub/ultralytics_yolov8_main
ENV YOLO_CONFIG_DIR=/tmp
RUN python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO model downloaded successfully')"

COPY . .

CMD ["python3", "main.py"]