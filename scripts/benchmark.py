"""
Performance Benchmark Suite

Validates model inference speed across different backends (PyTorch, ONNX, OpenVINO).
Runs two types of tests:
1. Static: Pure model inference time (theoretical max FPS).
2. Pipeline: Full system throughput including video decoding and preprocessing (real-world FPS).
"""
import cv2
import time
import numpy as np
from pathlib import Path
import json
import argparse
from threading import Thread
import queue

def benchmark_static(model_path, video_path, num_frames=200, imgsz=416, warmup=20):
    """
    Static Benchmark: Measures pure inference time
    - Reads frames from video
    - Measures ONLY inference time (no decode overhead in measurement)
    """
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"[STATIC] {model_path}")
    print(f"{'='*60}")
    
    # Load model
    try:
        if 'openvino' in str(model_path).lower():
            model = YOLO(model_path, task='detect')
        else:
            model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_width}x{video_height} → Inference: {imgsz}x{imgsz}")
    
    # Pre-read and resize frames
    print(f"Pre-loading {num_frames + warmup} frames...")
    frames = []
    for _ in range(num_frames + warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frames.append(cv2.resize(frame, (imgsz, imgsz)))
    cap.release()
    
    # Warmup
    print(f"Warming up ({warmup} frames)...")
    for i in range(warmup):
        model(frames[i], verbose=False)
    
    # Benchmark (pure inference)
    print(f"Benchmarking ({num_frames} frames)...")
    times = []
    for i in range(num_frames):
        t0 = time.perf_counter()
        model(frames[warmup + i], verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    
    print(f"  → Inference: {avg_ms:.1f} ms = {fps:.1f} FPS")
    
    return {
        'avg_ms': round(avg_ms, 2),
        'fps': round(fps, 1),
        'min_ms': round(min(times), 2),
        'max_ms': round(max(times), 2),
    }


def benchmark_pipeline(model_path, video_path, num_frames=200, imgsz=416, warmup=20):
    """
    Pipeline Benchmark: Measures full video processing
    - Multithreaded video decode
    - Resize + inference
    - Like real-time deploy.py
    """
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"[PIPELINE] {model_path}")
    print(f"{'='*60}")
    
    # Load model
    try:
        if 'openvino' in str(model_path).lower():
            model = YOLO(model_path, task='detect')
        else:
            model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Warmup
    print(f"Warming up...")
    for _ in range(warmup):
        ret, frame = cap.read()
        if ret:
            model(cv2.resize(frame, (imgsz, imgsz)), verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Threaded reader
    frame_queue = queue.Queue(maxsize=5)
    stop_flag = [False]
    
    def reader():
        while not stop_flag[0]:
            ret, frame = cap.read()
            if not ret:
                stop_flag[0] = True
                break
            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass
    
    reader_thread = Thread(target=reader, daemon=True)
    reader_thread.start()
    
    # Benchmark
    print(f"Benchmarking pipeline ({num_frames} frames)...")
    times = []
    count = 0
    
    while count < num_frames and not stop_flag[0]:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            break
        
        t0 = time.perf_counter()
        resized = cv2.resize(frame, (imgsz, imgsz))
        model(resized, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
        count += 1
    
    stop_flag[0] = True
    cap.release()
    
    if not times:
        return None
    
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    
    print(f"  → Pipeline: {avg_ms:.1f} ms = {fps:.1f} FPS")
    
    return {
        'avg_ms': round(avg_ms, 2),
        'fps': round(fps, 1),
        'min_ms': round(min(times), 2),
        'max_ms': round(max(times), 2),
    }


def main():
    parser = argparse.ArgumentParser(description='V3 Benchmark Suite')
    parser.add_argument('--video', default='test_video.mp4', help='Video file')
    parser.add_argument('--frames', type=int, default=200, help='Frames to test')
    parser.add_argument('--imgsz', type=int, default=416, help='Inference size')
    parser.add_argument('--mode', choices=['static', 'pipeline', 'both'], default='both')
    parser.add_argument('--output', default='reports/benchmark_v3.json', help='Output')
    args = parser.parse_args()
    
    models = {
        'PyTorch': 'models/best.pt',
        'ONNX': 'models/best.onnx',
        'OpenVINO_FP32': 'models/best_openvino_model/',
        'OpenVINO_INT8': 'models/best_int8_openvino_model/',
    }
    
    print("="*60)
    print("DRIVING NARRATOR - V3 BENCHMARK SUITE")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Mode: {args.mode}")
    print(f"Frames: {args.frames}")
    
    results = {'static': {}, 'pipeline': {}}
    
    for name, path in models.items():
        if not Path(path).exists():
            print(f"\n⚠️ Not found: {path}")
            continue
        
        if args.mode in ['static', 'both']:
            r = benchmark_static(path, args.video, args.frames, args.imgsz)
            if r:
                results['static'][name] = r
        
        if args.mode in ['pipeline', 'both']:
            r = benchmark_pipeline(path, args.video, args.frames, args.imgsz)
            if r:
                results['pipeline'][name] = r
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for mode in ['static', 'pipeline']:
        if results[mode]:
            print(f"\n[{mode.upper()}]")
            print(f"{'Model':<20} {'ms':<10} {'FPS':<10} {'Speedup':<10}")
            print("-"*50)
            baseline = None
            for name, r in results[mode].items():
                if baseline is None:
                    baseline = r['fps']
                    speedup = "1.0×"
                else:
                    speedup = f"{r['fps']/baseline:.1f}×"
                print(f"{name:<20} {r['avg_ms']:<10.1f} {r['fps']:<10.1f} {speedup:<10}")
    
    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to: {args.output}")

if __name__ == "__main__":
    main()
