import json
import traceback
import sys
import pytz
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from waggle.plugin import Plugin
from waggle.data.vision import Camera

from yolo_models import YOLOv8n, YOLOv5n, YOLOv10n


def get_chicago_time():
    """Get current time in Chicago timezone"""
    chicago_tz = pytz.timezone('America/Chicago')
    return datetime.now(chicago_tz).isoformat()


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


def run_detection_cycle_parallel(plugin, models, max_workers=3):
    """Run a single detection cycle with all models in parallel"""
    with Camera("bottom_camera") as camera:
        snapshot = camera.snapshot()
    
    timestamp = snapshot.timestamp
    
    snapshot_dt = datetime.fromtimestamp(timestamp / 1e9, tz=pytz.UTC)
    chicago_snapshot_time = snapshot_dt.astimezone(pytz.timezone('America/Chicago')).isoformat()
    
    all_results = {}
    cycle_start = time.time()
    
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
    
    parallel_duration = time.time() - cycle_start
    
    combined_data = {
        "image_timestamp_chicago": chicago_snapshot_time,
        "image_timestamp_ns": timestamp,
        "models_results": all_results,
        "parallel_execution_time_seconds": parallel_duration
    }
    plugin.publish("object.detections.all", json.dumps(combined_data), timestamp=timestamp)
    
    return timestamp, parallel_duration


def main():
    plugin_start_time = get_chicago_time()
    
    num_cores = multiprocessing.cpu_count()
    max_workers = min(3, num_cores, 3)
    
    with Plugin() as plugin:
        try:
            print(f"Initializing models with {max_workers} parallel workers...")
            models = {
                "YOLOv8n": YOLOv8n(),
                "YOLOv5n": YOLOv5n(),
                "YOLOv10n": YOLOv10n()
            }
            
            execution_times = []
            
            start_time = time.time()
            max_duration = (24 * 60 * 60) - 3
            interval = 3
            
            while (time.time() - start_time) < max_duration:
                timestamp, cycle_duration = run_detection_cycle_parallel(
                    plugin, models, max_workers
                )
                
                execution_times.append(cycle_duration)
                
                elapsed = time.time() - start_time
                next_cycle_time = ((int(elapsed / interval) + 1) * interval)
                sleep_time = next_cycle_time - elapsed
                
                if sleep_time > 0 and (elapsed + sleep_time) < max_duration:
                    time.sleep(sleep_time)
                else:
                    break
            
            plugin_finish_time = get_chicago_time()
            timing_summary = {
                "plugin_start_time_chicago": plugin_start_time,
                "plugin_finish_time_chicago": plugin_finish_time,
                "total_cycles": len(execution_times),
                "average_cycle_time_seconds": sum(execution_times) / len(execution_times) if execution_times else 0,
                "cycle_times_seconds": execution_times,
                "parallel_workers": max_workers,
                "cpu_cores_available": num_cores
            }
            
            plugin.publish("plugin.timing.summary", json.dumps(timing_summary))
            
        except Exception as e:
            error_data = {
                "status": "critical_error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            plugin.publish("plugin.error", json.dumps(error_data))
            
            print(f"Critical error in plugin: {e}", file=sys.stderr)
            traceback.print_exc()
            raise
    
    sys.exit(0)


if __name__ == "__main__":
    main()