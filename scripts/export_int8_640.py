"""
Export INT8 OpenVINO model at 640x640 input size
Keeps 416 version, creates new 640 version
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

# Load the FP32 model
model = YOLO('models/best.pt')

print("Exporting INT8 model at 640x640...")
print("This may take 10-15 minutes...\n")

# Export to OpenVINO INT8 with 640 input
export_path = model.export(
    format='openvino',
    imgsz=640,
    int8=True,
    data='LISA_stratified/data.yaml',  # For calibration
    half=False,
)

# Move to models folder with proper name
export_path = Path(export_path)
target_path = Path('models/best_int8_openvino_model_640')

if target_path.exists():
    shutil.rmtree(target_path)

shutil.move(str(export_path), str(target_path))

print(f"\n✅ Exported to: {target_path}")
print("\nYou now have:")
print("  - models/best_int8_openvino_model/     (416x416)")
print("  - models/best_int8_openvino_model_640/ (640x640)")
