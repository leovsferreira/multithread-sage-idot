from ultralytics import YOLO
import json

from waggle.plugin import Plugin
from waggle.data.vision import Camera


def detect_objects(image, model):
    results = model(image)
    
    detections = []
    for result in results:
        boxes = result.boxes
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
    model = YOLO("yolov8n.pt")
    
    with Plugin() as plugin:
        with Camera() as camera:
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

if __name__ == "__main__":
    main()