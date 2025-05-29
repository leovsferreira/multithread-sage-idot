from ultralytics import YOLO
import json
import traceback
import sys

from waggle.plugin import Plugin
from waggle.data.vision import Camera

def detect_objects(image, model):
    results = model(image)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
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
    with Plugin() as plugin:
        try:
            model_path = "/app/models/yolov8n.pt"
            model = YOLO(model_path)

            with Camera("bottom_camera") as camera:
                snapshot = camera.snapshot()

            timestamp = snapshot.timestamp

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

            plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)

        except Exception as e:
            try:
                error_timestamp = timestamp
            except NameError:
                import time
                error_timestamp = time.time_ns()

            error_data = {
                "status": "error",
                "error_type": type(e).name,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }

            plugin.publish("plugin.error", json.dumps(error_data), timestamp=error_timestamp)

            print(f"Error in plugin: {e}", file=sys.stderr)
            traceback.print_exc()

            raise

if __name__ == "__main__":
    main()