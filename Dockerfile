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

RUN python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded successfully')"

COPY . .

CMD ["python3", "main.py"]