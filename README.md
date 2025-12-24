# Driving Narrator

> **Real-Time Traffic Sign Detection** | **14 FPS** | **97% mAP** | **Legacy Edge Hardware**

High-performance computer vision system engineered for legacy CPU hardware (Intel i5 Broadwell).

## System Capabilities

* **⚡ Real-Time Inference (14 FPS):**
    Runs at 14 FPS on 720p video using OpenVINO INT8 quantization and a custom multi-threaded pipeline, delivering a **6x speedup** over standard PyTorch.

* **🎯 High-Precision Model (97% mAP):**
    Trained a custom YOLOv11 Nano model on 15,876 images (LISA Dataset), achieving **97.1% mAP@0.5** on held-out test sets to validate robust generalization.

* **📉 Optimized Efficiency:**
    Reduced model footprint by **38%** (5.2 MB → 3.2 MB) enabling deployment on 15W TDP hardware without GPUs.

---

## Speed vs Accuracy Trade-off

![Accuracy vs Speed](reports/speed_vs_accuracy_tradeoff.png)

*The dual-mode system allows switching between Performance Mode (14 FPS) and Precision Mode (97% mAP) based on use case.*

---

## Hardware Tested

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i5-5250U (Broadwell, 5th Gen) |
| Cores | 2 Cores / 4 Threads |
| TDP | 15W |
| RAM | 8 GB |
| OS | macOS Monterey |

---

## Project Structure

```
├── src/                    # Core detection modules
├── scripts/                # Deployment and benchmarking
├── models/                 # Trained weights (PT, ONNX, INT8)
├── tests/                  # Unit tests
├── reports/                # Benchmark results and charts

├── .github/                # CI workflows
├── Dockerfile              # Container deployment
└── requirements.txt        # Dependencies
```

---

## Technical Implementation

### 1. The Multi-Threaded Pipeline

Implements a producer-consumer pattern to solve the "I/O Blocking" problem common in Python.

```
[Capture Thread] --(Frames)--> [Queue] --(Frames)--> [Inference Thread]
       |                                                    |
  Running at 30Hz                                    Running at 14Hz
```

* **Result:** Decouples video decoding from model inference, stabilizing latency at **40ms**.

### 2. Dual-Mode Inference Configuration

To handle the trade-off between speed and accuracy on legacy hardware, the system supports two run-time modes:

| Configuration | FPS | Accuracy | Use Case |
|:--- |:---:|:---:|:--- |
| **🚀 Performance Mode (Default)** | **14.0** | **93.2%** | **Driving Safety (Low Latency)** |
| 🎯 Precision Mode | 7.4 | 97.1% | Offline Analysis |

*Note: Performance Mode uses 416x416 resolution input, while Precision Mode uses 640x640.*

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Sasaank79/Driving-Narrator-Legacy-Edge.git
cd Driving-Narrator-Legacy-Edge

# Install dependencies
pip install -r requirements.txt

# Run in Performance Mode (Default: 416px / 14 FPS)
python scripts/deploy_720p.py --video path/to/video.mp4

# Run in Precision Mode (640px / 97% mAP)
python scripts/deploy_720p.py --video path/to/video.mp4 --imgsz 640
```

---

## Benchmarks (Intel i5-5250U)

| Metric | Backend | FPS | Speedup | Notes |
|--------|---------|-----|---------|-------|
| **Raw Inference** | PyTorch FP32 | 4.1 | 1x | Baseline |
| | OpenVINO INT8 | 25.1 | **~6.1x** | Validates 6x Efficiency |
| **Full Pipeline** | PyTorch (App) | 3.8 | 1x | Includes I/O overhead |
| | OpenVINO (App) | 14.0 | ~3.7x | Real-World Performance |

---

## Dataset & Training

- **Dataset:** LISA Traffic Sign Dataset (47 Classes, Stratified Split)
- **Training Hardware:** NVIDIA T4 (Google Colab)
- **Training Strategy:** 50 Epochs, SGD Optimizer, Mosaic Augmentation

---

## License

**AGPL-3.0** (inherited from Ultralytics YOLOv11).

This project uses the **LISA Traffic Sign Dataset** from the Laboratory for Intelligent and Safe Automobiles (UCSD).
