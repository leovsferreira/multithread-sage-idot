from ultralytics import YOLO
import os
import time


class YOLOModel:
    """Base class for YOLO model handling"""
    
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model = YOLO(self.model_path)
    
    def detect(self, image):
        """Run detection on an image"""
        start_time = time.time()
        results = self.model(image)
        inference_time = time.time() - start_time
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls.item())
                    cls_name = self.model.names[cls]
                    conf = box.conf.item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "model": self.model_name,
            "detections": detections,
            "counts": class_counts,
            "total_objects": len(detections),
            "inference_time_seconds": inference_time
        }


class YOLOv8n(YOLOModel):
    """YOLOv8n model handler"""
    
    def __init__(self):
        super().__init__("YOLOv8n", "/app/models/yolov8n.pt")


class YOLOv5n(YOLOModel):
    """YOLOv5n model handler"""
    
    def __init__(self):
        super().__init__("YOLOv5n", "/app/models/yolov5nu.pt")


class YOLOv10n(YOLOModel):
    """YOLOv10n model handler"""
    
    def __init__(self):
        super().__init__("YOLOv10n", "/app/models/yolov10n.pt")