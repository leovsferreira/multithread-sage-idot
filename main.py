import json
import traceback
import sys
import os
import pytz
import time
from datetime import datetime

from waggle.plugin import Plugin
from waggle.data.vision import Camera

from yolo_models import YOLOv8n, YOLOv5n, YOLOv10n


def get_chicago_time():
    """Get current time in Chicago timezone"""
    chicago_tz = pytz.timezone('America/Chicago')
    return datetime.now(chicago_tz).isoformat()


def run_detection_cycle(plugin, models):
    """Run a single detection cycle with all models"""
    with Camera("bottom_camera") as camera:
        snapshot = camera.snapshot()
    
    timestamp = snapshot.timestamp
    
    snapshot_dt = datetime.fromtimestamp(timestamp / 1e9, tz=pytz.UTC)
    chicago_snapshot_time = snapshot_dt.astimezone(pytz.timezone('America/Chicago')).isoformat()
    
    image_filename = f"detection_image_{int(timestamp)}.jpg"
    snapshot.save(image_filename)
    plugin.upload_file(image_filename, timestamp=timestamp)
    
    all_results = {}
    for model_name, model_instance in models.items():
        try:
            detection_result = model_instance.detect(snapshot.data)
            all_results[model_name] = detection_result
            
            plugin.publish(
                f"object.detections.{model_name.lower()}", 
                json.dumps(detection_result), 
                timestamp=timestamp
            )
        except Exception as e:
            error_data = {
                "model": model_name,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            plugin.publish(
                f"model.error.{model_name.lower()}", 
                json.dumps(error_data), 
                timestamp=timestamp
            )
            print(f"Error in {model_name}: {e}", file=sys.stderr)
    
    combined_data = {
        "image_timestamp_chicago": chicago_snapshot_time,
        "image_timestamp_ns": timestamp,
        "models_results": all_results
    }
    plugin.publish("object.detections.all", json.dumps(combined_data), timestamp=timestamp)
    
    try:
        os.remove(image_filename)
    except:
        pass
    
    return timestamp


def main():
    plugin_start_time = get_chicago_time()
    
    with Plugin() as plugin:
        try:
            models = {
                "YOLOv8n": YOLOv8n(),
                "YOLOv5n": YOLOv5n(),
                "YOLOv10n": YOLOv10n()
            }
            
            execution_times = []
            
            start_time = time.time()
            max_duration = 55 
            interval = 20
            
            while (time.time() - start_time) < max_duration:
                cycle_start = time.time()
                
                run_detection_cycle(plugin, models)
                
                cycle_duration = time.time() - cycle_start
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
                "cycle_times_seconds": execution_times
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