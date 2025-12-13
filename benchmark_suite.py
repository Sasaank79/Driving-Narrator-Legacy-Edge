"""
Comprehensive Benchmark: All models, with/without multithreading
Generates data for documentation updates
"""
from ultralytics import YOLO
import time
import cv2
from threading import Thread
import queue
import json

print("=" * 70)
print("COMPREHENSIVE BENCHMARK: Single-Threaded vs Multithreaded")
print("=" * 70)

# Test configurations
MODELS = {
    'PyTorch FP32': 'models/best.pt',
    'ONNX': 'models/best.onnx', 
    'OpenVINO FP16': 'models/best_openvino_model/',
    'OpenVINO INT8': 'models/best_int8_openvino_model/',
}

VIDEO_PATH = 'test_video.mp4'
WARMUP = 20
FRAMES = 100

# Load test image
test_img = cv2.imread('reports/demo_proof/keepRight_1323813131-avi_image1_png.rf.189939290f3e41e75dcad5d711ab56ce.jpg')
test_img_416 = cv2.resize(test_img, (416, 416))

results_data = {}

# === SINGLE IMAGE BENCHMARK ===
print("\n[PART 1] Single Image Benchmark (Raw Inference)")
print("-" * 70)

for name, path in MODELS.items():
    print(f"  Loading {name}...")
    if 'openvino' in path.lower():
        model = YOLO(path, task='detect')
    else:
        model = YOLO(path)
    
    # Warmup
    for _ in range(WARMUP):
        model(test_img_416, verbose=False)
    time.sleep(0.3)
    
    # Benchmark
    times = []
    for _ in range(FRAMES):
        start = time.time()
        model(test_img_416, verbose=False)
        times.append(time.time() - start)
    
    fps = FRAMES / sum(times)
    results_data[f'{name}_image'] = round(fps, 1)
    print(f"  {name:20s}: {fps:.1f} FPS (raw inference)")

# === VIDEO BENCHMARK: SINGLE-THREADED ===
print("\n[PART 2] Video Benchmark - Single-Threaded")
print("-" * 70)

for name, path in MODELS.items():
    print(f"  Loading {name}...")
    if 'openvino' in path.lower():
        model = YOLO(path, task='detect')
    else:
        model = YOLO(path)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Warmup
    for _ in range(WARMUP):
        ret, f = cap.read()
        if ret:
            model(cv2.resize(f, (416, 416)), verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    time.sleep(0.3)
    
    # Benchmark (with viz, no display)
    times = []
    for _ in range(FRAMES):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        start = time.time()
        resized = cv2.resize(frame, (416, 416))
        result = model(resized, verbose=False)
        annotated = result[0].plot()
        times.append(time.time() - start)
    
    cap.release()
    fps = FRAMES / sum(times)
    results_data[f'{name}_video_single'] = round(fps, 1)
    print(f"  {name:20s}: {fps:.1f} FPS (video, single-threaded)")

# === VIDEO BENCHMARK: MULTITHREADED ===
print("\n[PART 3] Video Benchmark - Multithreaded")
print("-" * 70)

for name, path in MODELS.items():
    print(f"  Loading {name}...")
    if 'openvino' in path.lower():
        model = YOLO(path, task='detect')
    else:
        model = YOLO(path)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_queue = queue.Queue(maxsize=5)
    stop_flag = False
    
    def reader():
        while not stop_flag:
            ret, f = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                frame_queue.put(f, timeout=0.05)
            except queue.Full:
                pass
    
    # Warmup (single-threaded first)
    for _ in range(WARMUP):
        ret, f = cap.read()
        if ret:
            model(cv2.resize(f, (416, 416)), verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    time.sleep(0.3)
    
    # Start reader thread
    t = Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.2)
    
    # Benchmark
    times = []
    for _ in range(FRAMES):
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        start = time.time()
        resized = cv2.resize(frame, (416, 416))
        result = model(resized, verbose=False)
        annotated = result[0].plot()
        times.append(time.time() - start)
    
    stop_flag = True
    cap.release()
    
    fps = FRAMES / sum(times) if times else 0
    results_data[f'{name}_video_multi'] = round(fps, 1)
    print(f"  {name:20s}: {fps:.1f} FPS (video, multithreaded)")

# === SUMMARY ===
print("\n" + "=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<20} {'Image':<10} {'Video ST':<12} {'Video MT':<12} {'Improvement':<12}")
print("-" * 70)

for name in MODELS.keys():
    img = results_data.get(f'{name}_image', 0)
    st = results_data.get(f'{name}_video_single', 0)
    mt = results_data.get(f'{name}_video_multi', 0)
    improvement = ((mt - st) / st * 100) if st > 0 else 0
    print(f"{name:<20} {img:<10.1f} {st:<12.1f} {mt:<12.1f} {'+' if improvement > 0 else ''}{improvement:.0f}%")

# Calculate speedups
pt_mt = results_data.get('PyTorch FP32_video_multi', 1)
int8_mt = results_data.get('OpenVINO INT8_video_multi', 1)
speedup = int8_mt / pt_mt if pt_mt > 0 else 0

print("-" * 70)
print(f"\nSpeedup (INT8 vs PyTorch): {speedup:.1f}×")
print(f"Best INT8 Video FPS: {int8_mt} (multithreaded)")

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
print("\nResults saved to benchmark_results.json")
