# Trained Models

## Available Models

| Model | Format | Size | mAP@0.5 | Use Case |
|-------|--------|------|---------|----------|
| `best.pt` | PyTorch | 10.8 MB | 94.8% | Training, fine-tuning |
| `best.onnx` | ONNX | 10.5 MB | 94.8% | Cross-platform inference |
| `best_openvino_model/` | OpenVINO FP16 | 5.4 MB | 94.8% | Intel CPU inference |
| `best_int8_openvino_model/` | OpenVINO INT8 | 3.2 MB | 93.2% | **Production (fastest)** |

## Model Architecture
- **Base:** YOLOv11 Nano (Ultralytics)
- **Parameters:** 2.6M
- **Backbone:** CSPDarknet
- **Neck:** PANet
- **Input Size:** 416×416 (inference)

## Training Configuration
- **Platform:** Google Colab T4 GPU
- **Epochs:** 25 (best at epoch 22)
- **Batch Size:** 32
- **Optimizer:** SGD (lr=0.01, cosine decay)
- **Early Stopping:** Patience=10

## Quantization Details
- **Method:** Post-Training Quantization (PTQ)
- **Calibration:** 200 images from training set
- **Framework:** OpenVINO NNCF
- **Accuracy Drop:** 1.6% (94.8% → 93.2%)

## Inference Performance (Intel i5-5250U, CPU-only)

| Model | Raw FPS | Pipeline FPS | Speedup |
|-------|---------|--------------|---------|
| PyTorch FP32 | ~4 | 2.3 | 1.0× |
| ONNX | ~8 | 9.3 | 4.0× |
| OpenVINO FP16 | ~4 | 4.3 | 1.9× |
| **OpenVINO INT8** | **~24** | **13.9** | **6.0×** |

## Usage
```python
from ultralytics import YOLO

# PyTorch
model = YOLO("models/best.pt")

# OpenVINO INT8
model = YOLO("models/best_int8_openvino_model")

results = model("image.jpg", imgsz=416, conf=0.25)
```
