import json
import traceback
import sys
import pytz
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from collections import deque

from waggle.plugin import Plugin
from waggle.data.vision import Camera

from yolo_models import YOLOv8n, YOLOv5n, YOLOv10n


def run_model_detection(model_name, model_instance, image_data):
    """Run detection for a single model - used for parallel execution"""
    try:
        detection_result = model_instance.detect(image_data)
        return model_name, detection_result, None
    except Exception as e:
        return model_name, None, {
            "model": model_name,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


def run_detection_cycle_parallel(models, max_workers=3):
    """Run a single detection cycle with all models in parallel"""
    with Camera("bottom_camera") as camera:
        snapshot = camera.snapshot()
    
    timestamp = snapshot.timestamp
    all_results = {}
    errors = []
    
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
            elif error:
                errors.append(error)
    
    return timestamp, all_results, errors


def batch_publish_results(plugin, results_queue):
    """Publish all queued results in batch"""
    while results_queue:
        timestamp, results, errors = results_queue.popleft()
        
        # Convert timestamp for publishing
        snapshot_dt = datetime.fromtimestamp(timestamp / 1e9, tz=pytz.UTC)
        chicago_snapshot_time = snapshot_dt.astimezone(pytz.timezone('America/Chicago')).isoformat()
        
        # Publish detection results
        combined_data = {
            "image_timestamp_chicago": chicago_snapshot_time,
            "image_timestamp_ns": timestamp,
            "models_results": results
        }
        plugin.publish("object.detections.all", json.dumps(combined_data), timestamp=timestamp)
        
        # Publish any errors
        for error in errors:
            plugin.publish(
                f"model.error.{error['model'].lower()}", 
                json.dumps(error), 
                timestamp=timestamp
            )


def main():
    num_cores = multiprocessing.cpu_count()
    max_workers = min(3, num_cores)
    
    with Plugin() as plugin:
        try:
            # Initialize models once
            models = {
                "YOLOv8n": YOLOv8n(),
                "YOLOv5n": YOLOv5n(),
                "YOLOv10n": YOLOv10n()
            }
            
            start_time = time.time()
            max_duration = 57  # 3 second buffer for publishing
            results_queue = deque()
            
            cycle_count = 0
            
            # Run inference cycles as fast as possible
            while (time.time() - start_time) < max_duration:
                timestamp, results, errors = run_detection_cycle_parallel(
                    models, max_workers
                )
                
                # Queue results instead of publishing immediately
                results_queue.append((timestamp, results, errors))
                cycle_count += 1
            
            # Batch publish all results at the end
            print(f"Completed {cycle_count} inference cycles. Publishing results...", file=sys.stderr)
            publish_start = time.time()
            batch_publish_results(plugin, results_queue)
            publish_time = time.time() - publish_start
            
            # Publish summary
            summary_data = {
                "total_cycles": cycle_count,
                "total_runtime_seconds": time.time() - start_time,
                "publish_time_seconds": publish_time,
                "timestamp_chicago": datetime.now(pytz.timezone('America/Chicago')).isoformat()
            }
            plugin.publish("inference.summary", json.dumps(summary_data))
            
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


if __name__ == "__main__":
    main()