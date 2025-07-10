import json
import traceback
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from waggle.plugin import Plugin
from waggle.data.vision import Camera

def should_publish_image():
    """Check if current minute is a multiple of 5"""
    current_minute = datetime.now().minute
    return current_minute % 5 == 0


def run_model_detection(model_name, model_instance, image_data):
    """Run detection for a single model - used for parallel execution"""
    try:
        detection_result = model_instance.detect(image_data)
        return model_name, detection_result, None
    except Exception as e:
        error_data = {
            "model": model_name,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        print(f"Error in {model_name}: {e}", file=sys.stderr)
        return model_name, None, error_data


def run_detection_cycle_parallel(plugin, models, max_workers=3, publish_image=False):
    """Run a single detection cycle with all models in parallel"""
    with Camera("bottom_camera") as camera:
        snapshot = camera.snapshot()
    
    timestamp = snapshot.timestamp
    
    if publish_image:
        snapshot.save("snapshot.jpg")
        plugin.upload_file("snapshot.jpg", timestamp=timestamp)
    
    all_results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                run_model_detection, 
                model_name, 
                model_instance, 
                snapshot.data
            ): model_name 
            for model_name, model_instance in models.items()
        }
        
        for future in as_completed(future_to_model):
            model_name, result, error = future.result()
            if result is not None:
                all_results[model_name] = result
            else:
                plugin.publish(
                    f"model.error.{model_name.lower()}", 
                    json.dumps(error), 
                    timestamp=timestamp
                )
    
    combined_data = {
        "image_timestamp_ns": timestamp,
        "models_results": all_results
    }
    plugin.publish("object.detections.all", json.dumps(combined_data), timestamp=timestamp)
    
    return timestamp


def main():
    start_time = time.time()
    max_duration = 58
    
    from yolo_models import YOLOv8n, YOLOv5n, YOLOv10n
    
    num_cores = multiprocessing.cpu_count()
    max_workers = min(3, num_cores, 3)
    
    with Plugin() as plugin:
        try:
            models = {
                "YOLOv8n": YOLOv8n(),
                "YOLOv5n": YOLOv5n(),
                "YOLOv10n": YOLOv10n()
            }
            
            image_published = False
            
            while (time.time() - start_time) < max_duration:
                if should_publish_image() and not image_published:
                    publish_image = True
                    image_published = True
                else:
                    publish_image = False
                
                timestamp = run_detection_cycle_parallel(
                    plugin, models, max_workers, 
                    publish_image=publish_image
                )
            
        except Exception as e:
            error_data = {
                "status": "critical_error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            plugin.publish("plugin.error", json.dumps(error_data))
            raise
    
    sys.exit(0)


if __name__ == "__main__":
    main()