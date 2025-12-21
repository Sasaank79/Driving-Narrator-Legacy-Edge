"""
Quick evaluate INT8 640 model on test set
Minimal output - just the final numbers
"""
from ultralytics import YOLO
import json

print("Loading INT8 640 model...")
model = YOLO('models/best_int8_openvino_model_640/', task='detect')

print("Evaluating on test set (this takes a few minutes)...\n")

metrics = model.val(
    data='LISA_stratified/data.yaml',
    split='test',
    imgsz=640,
    verbose=False  # Quiet output
)

print("="*50)
print("INT8 640 MODEL - TEST SET RESULTS")
print("="*50)
print(f"mAP@0.5:      {metrics.box.map50:.2%}")
print(f"mAP@0.5:0.95: {metrics.box.map:.2%}")
print(f"Precision:    {metrics.box.mp:.2%}")
print(f"Recall:       {metrics.box.mr:.2%}")
print("="*50)

# Also do val set
print("\nEvaluating on val set...")
metrics_val = model.val(
    data='LISA_stratified/data.yaml',
    split='val',
    imgsz=640,
    verbose=False
)

print("\n" + "="*50)
print("FULL COMPARISON (V3 METRICS)")
print("="*50)
print(f"{'Model':<20} {'Split':<8} {'mAP@0.5':<12} {'Size':<8}")
print("-"*50)
print(f"{'FP32 (best.pt)':<20} {'val':<8} {'96.15%':<12} {'5.2 MB':<8}")
print(f"{'FP32 (best.pt)':<20} {'test':<8} {'96.86%':<12} {'5.2 MB':<8}")
print(f"{'INT8 (416)':<20} {'val':<8} {'92.36%':<12} {'3.2 MB':<8}")
print(f"{'INT8 (416)':<20} {'test':<8} {'93.19%':<12} {'3.2 MB':<8}")
print(f"{'INT8 (640)':<20} {'val':<8} {metrics_val.box.map50:<12.2%} {'TBD':<8}")
print(f"{'INT8 (640)':<20} {'test':<8} {metrics.box.map50:<12.2%} {'TBD':<8}")
print("="*50)

# Quantization loss
loss_640 = 0.9686 - metrics.box.map50
loss_416 = 0.9686 - 0.9319
print(f"\nQuantization loss (FP32→INT8):")
print(f"  At 640: {loss_640:.2%}")
print(f"  At 416: {loss_416:.2%}")

# Save
results = {
    'INT8_640_test': {
        'mAP50': round(float(metrics.box.map50), 4),
        'mAP50_95': round(float(metrics.box.map), 4),
    },
    'INT8_640_val': {
        'mAP50': round(float(metrics_val.box.map50), 4),
        'mAP50_95': round(float(metrics_val.box.map), 4),
    }
}
with open('reports/int8_640_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved to: reports/int8_640_results.json")
