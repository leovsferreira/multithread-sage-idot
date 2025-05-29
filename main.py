from ultralytics import YOLO
import json
import traceback
import sys
import time
import os
import signal
from contextlib import contextmanager

from waggle.plugin import Plugin
from waggle.data.vision import Camera

@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def detect_objects(image, model):
    results = model(image)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:  # Check if boxes exist
            for box in boxes:
                cls = int(box.cls.item())
                cls_name = model.names[cls]
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
    
    return detections

def main():
    plugin = None
    try:
        plugin = Plugin()
        
        # Check if model file exists locally
        model_path = "/app/models/yolov8n.pt"
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}", file=sys.stderr)
            print("Available files in /app:", os.listdir("/app"), file=sys.stderr)
            if os.path.exists("/app/models"):
                print("Files in /app/models:", os.listdir("/app/models"), file=sys.stderr)
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        
        print(f"Loading YOLO model from {model_path}...", flush=True)
        with timeout(30):  # 30 second timeout for model loading
            model = YOLO(model_path)
        print("Model loaded successfully", flush=True)
        
        # Debug: List available cameras
        try:
            from waggle.data.vision import list_cameras
            available_cameras = list_cameras()
            print(f"Available cameras: {available_cameras}", flush=True)
        except Exception as e:
            print(f"Could not list cameras: {e}", flush=True)
        
        print("Initializing camera...", flush=True)
        with timeout(10):  # 10 second timeout for camera
            with Camera("bottom_camera") as camera:
                print("Taking snapshot...", flush=True)
                snapshot = camera.snapshot()
                timestamp = snapshot.timestamp
                print(f"Snapshot taken at {timestamp}", flush=True)
        
        print("Running object detection...", flush=True)
        with timeout(30):  # 30 second timeout for detection
            detections = detect_objects(snapshot.data, model)
        
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detection_data = {
            "detections": detections,
            "counts": class_counts,
            "total_objects": len(detections)
        }
        
        print(f"Publishing {len(detections)} detections", flush=True)
        plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)
        print("Data published successfully", flush=True)
        
    except TimeoutError as e:
        error_timestamp = time.time_ns()
        error_data = {
            "status": "timeout_error",
            "error_type": "TimeoutError",
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        
        if plugin:
            plugin.publish("plugin.error", json.dumps(error_data), timestamp=error_timestamp)
        
        print(f"Timeout error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        try:
            error_timestamp = timestamp
        except NameError:
            error_timestamp = time.time_ns()
        
        error_data = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        
        if plugin:
            plugin.publish("plugin.error", json.dumps(error_data), timestamp=error_timestamp)
        
        print(f"Error in plugin: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if plugin:
            plugin.close()

if __name__ == "__main__":
    main()