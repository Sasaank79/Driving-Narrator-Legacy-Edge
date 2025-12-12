# 🚗 Driving Narrator

**Real-Time Traffic Sign Detection on Legacy Hardware — No GPU Required**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-INT8-green.svg)](https://docs.openvino.ai)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11_Nano-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 🎯 What This Project Proves

**A 10-year-old laptop with no GPU can run a state-of-the-art object detector at real-time speeds.**

| Metric | Achieved |
|--------|----------|
| **Inference Speed** | 13 FPS @ 720p |
| **Accuracy** | 93.2% mAP@0.5 |
| **Model Size** | 3.2 MB (INT8) |
| **Hardware** | Intel i5-5250U, 8GB RAM, No GPU |

This challenges the industry narrative that edge AI requires specialized accelerators (NPUs/TPUs).

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Curation  │ ──▶ │  Cloud Training │ ──▶ │  Quantization   │ ──▶ │ Edge Deployment │
│   (Roboflow)    │     │  (Colab T4 GPU) │     │   (OpenVINO)    │     │  (MacBook Air)  │
│   16K+ images   │     │   YOLOv11 Nano  │     │   FP32 → INT8   │     │   13 FPS CPU    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Sasaank79/Driving-Narrator-Legacy-Edge.git
cd Driving-Narrator-Legacy-Edge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference

```bash
python3 deploy.py
```

This runs real-time detection on the included `test_video.mp4`. Press `q` to quit.

### 3. Try Your Own Video

```python
# In deploy.py, change:
VIDEO_PATH = "your_video.mp4"
```

---

## 📁 Project Structure

```
Driving-Narrator-Legacy-Edge/
├── models/
│   ├── best.pt                    # PyTorch weights (10 MB)
│   ├── best.onnx                  # ONNX export (10 MB)
│   ├── best_openvino_model/       # OpenVINO FP16 (5.4 MB)
│   └── best_int8_openvino_model/  # OpenVINO INT8 (3.2 MB) ⭐
├── reports/
│   ├── results_custom.png         # Training curves
│   ├── confusion_matrix.png       # Class performance
│   └── demo_proof/                # Detection samples
├── benchmarks/
│   └── lisa_benchmark_results.txt # FPS measurements
├── research/
│   ├── benchmark_cpu.py           # PyTorch benchmark
│   ├── benchmark_onnx.py          # ONNX benchmark
│   └── benchmark_openvino.py      # OpenVINO benchmark
├── Whitepaper/
│   └── V5.md                      # Technical whitepaper
├── deploy.py                      # Main inference script ⭐
├── app_video.py                   # Video file inference
├── app_webcam.py                  # Webcam inference
├── benchmark_lisa.py              # Full benchmark suite
├── LISA_Training.ipynb            # Colab training notebook
├── test_video.mp4                 # Sample dash-cam video (44 MB)
└── requirements.txt
```

---

## 📊 Performance Benchmarks

Tested on MacBook Air 2015 (Intel i5-5250U, 8GB RAM, Intel HD 6000):

| Format | mAP@0.5 | FPS | Size | Speedup |
|--------|---------|-----|------|---------|
| PyTorch FP32 | 94.8% | 6.6 | 10.3 MB | 1.0× |
| ONNX | 94.8% | 11.2 | 10.1 MB | 1.7× |
| OpenVINO FP16 | 94.8% | 6.4 | 5.4 MB | 1.0× |
| **OpenVINO INT8** | **93.2%** | **13.0** | **3.2 MB** | **2.0×** |

---

## 🔧 Training (Optional)

Training was done on Google Colab with a T4 GPU. To reproduce:

1. Open `LISA_Training.ipynb` in [Google Colab](https://colab.research.google.com)
2. Download the [LISA dataset from Roboflow](https://universe.roboflow.com/suryaworkspace/lisa-road-signs-ftd0q/dataset/2)
3. Run all cells

**Training Config:**
- Model: YOLOv11 Nano (2.6M params)
- Epochs: 25 (early stopping at 22)
- Batch: 32
- Image Size: 640×640 train / 416×416 inference

---

## 📚 Dataset

**LISA Traffic Sign Dataset** — 47 US MUTCD sign classes

- **Source:** [Roboflow Universe](https://universe.roboflow.com/suryaworkspace/lisa-road-signs-ftd0q/dataset/2)
- **Original:** [LISA @ UC San Diego](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html)
- **Augmented:** 16,544 images (brightness, rotation, shear, occlusion)

---

## 📝 Whitepaper

Full technical details in [`Whitepaper/V5.md`](Whitepaper/V5.md):
- Methodology & system architecture
- Quantization pipeline (FP32 → INT8)
- Failure mode analysis
- Engineering trade-offs

---

## 🎓 About This Project

This is an independent post-graduation R&D project (June–December 2025) demonstrating:

1. **End-to-end ML pipeline:** Data curation → Training → Optimization → Deployment
2. **Constraint-first engineering:** Optimizing for a specific hardware target
3. **Production-grade practices:** Honest benchmarks, documented trade-offs, reproducible results

**Author:** [Surya Sasaank Yanamandra](https://www.linkedin.com/in/surya-sasaank-yanamandra/)

---

## 📄 License

MIT License — Free for educational and commercial use.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Intel OpenVINO](https://docs.openvino.ai) for INT8 quantization
- [Roboflow](https://roboflow.com) for dataset tools
- [Google Colab](https://colab.research.google.com) for free GPU training
