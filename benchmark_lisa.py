"""
LISA Model Benchmark - Compare PyTorch vs ONNX vs OpenVINO (FP16 & INT8)
Tests all model formats at 416x416 on CPU.
"""

import time
import os
import numpy as np

# --- CONFIGURATION ---
MODELS_DIR = "models"
IMG_SIZE_PT_ONNX = 416      # PyTorch/ONNX can use any size
IMG_SIZE_OPENVINO = 640     # OpenVINO models were exported at 640
NUM_WARMUP = 5
NUM_RUNS = 50

def get_model_size(path):
    """Get model size in MB."""
    if os.path.isdir(path):
        total = sum(os.path.getsize(os.path.join(path, f)) 
                   for f in os.listdir(path) if f.endswith(('.bin', '.xml', '.onnx')))
        return total / (1024 * 1024)
    elif os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

def benchmark_pytorch(model_path, img_size, num_warmup, num_runs):
    """Benchmark PyTorch model."""
    from ultralytics import YOLO
    import torch
    
    print("\n" + "=" * 60)
    print(f"PyTorch Benchmark ({img_size}x{img_size})")
    print("=" * 60)
    
    size_mb = get_model_size(model_path)
    print(f"Model: {model_path} ({size_mb:.1f} MB)")
    
    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        model(dummy, imgsz=img_size, verbose=False)
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        model(dummy, imgsz=img_size, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
    
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"\nResult: {avg_ms:.1f}ms, {fps:.1f} FPS")
    
    return {'format': 'PyTorch', 'avg_ms': avg_ms, 'fps': fps, 'size_mb': size_mb}


def benchmark_onnx(model_path, img_size, num_warmup, num_runs):
    """Benchmark ONNX Runtime."""
    import onnxruntime as ort
    
    print("\n" + "=" * 60)
    print(f"ONNX Runtime Benchmark ({img_size}x{img_size})")
    print("=" * 60)
    
    size_mb = get_model_size(model_path)
    print(f"Model: {model_path} ({size_mb:.1f} MB)")
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"⚠️ ONNX load failed: {e}")
        print("   Tip: Model was exported with opset 22, but onnxruntime supports up to 21.")
        print("   Run: pip install --upgrade onnxruntime")
        return None
    
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    
    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
    
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"\nResult: {avg_ms:.1f}ms, {fps:.1f} FPS")
    
    return {'format': 'ONNX', 'avg_ms': avg_ms, 'fps': fps, 'size_mb': size_mb}


def benchmark_openvino(model_dir, label, num_warmup, num_runs):
    """Benchmark OpenVINO (auto-detects input size from model)."""
    from openvino import Core
    
    xml_path = os.path.join(model_dir, "best.xml")
    if not os.path.exists(xml_path):
        print(f"❌ Model not found: {xml_path}")
        return None
    
    size_mb = get_model_size(model_dir)
    
    try:
        core = Core()
        model = core.read_model(xml_path)
        compiled = core.compile_model(model, "CPU")
        infer_request = compiled.create_infer_request()
        input_layer = compiled.input(0)
        
        # Auto-detect input size from model
        input_shape = input_layer.shape
        img_size = input_shape[2]  # [batch, channels, height, width]
        
        print("\n" + "=" * 60)
        print(f"OpenVINO {label} Benchmark ({img_size}x{img_size})")
        print("=" * 60)
        print(f"Model: {model_dir} ({size_mb:.1f} MB)")
        
        dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        
        # Warmup
        print(f"Warmup ({num_warmup} runs)...")
        for _ in range(num_warmup):
            infer_request.infer({input_layer: dummy})
        
        # Benchmark
        print(f"Benchmarking ({num_runs} runs)...")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            infer_request.infer({input_layer: dummy})
            times.append((time.perf_counter() - start) * 1000)
            if (i + 1) % 10 == 0:
                print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
        
        avg_ms = np.mean(times)
        fps = 1000 / avg_ms
        print(f"\nResult: {avg_ms:.1f}ms, {fps:.1f} FPS")
        
        return {'format': f'OpenVINO {label}', 'avg_ms': avg_ms, 'fps': fps, 'size_mb': size_mb, 'img_size': img_size}
    
    except Exception as e:
        print(f"⚠️ OpenVINO {label} failed: {e}")
        return None


def main():
    print("=" * 70)
    print("LISA Traffic Sign Model Benchmark")
    print("Comparing: PyTorch vs ONNX vs OpenVINO FP16 vs OpenVINO INT8")
    print("=" * 70)
    
    results = []
    
    # 1. PyTorch (416x416)
    pt_path = f"{MODELS_DIR}/best.pt"
    if os.path.exists(pt_path):
        results.append(benchmark_pytorch(pt_path, IMG_SIZE_PT_ONNX, NUM_WARMUP, NUM_RUNS))
    else:
        print(f"⚠️ Skipping PyTorch: {pt_path} not found")
    
    # 2. ONNX (416x416)
    onnx_path = f"{MODELS_DIR}/best.onnx"
    if os.path.exists(onnx_path):
        r = benchmark_onnx(onnx_path, IMG_SIZE_PT_ONNX, NUM_WARMUP, NUM_RUNS)
        if r: results.append(r)
    else:
        print(f"⚠️ Skipping ONNX: {onnx_path} not found")
    
    # 3. OpenVINO FP16 (auto-detects size)
    fp16_dir = f"{MODELS_DIR}/best_openvino_model"
    if os.path.exists(fp16_dir):
        r = benchmark_openvino(fp16_dir, "FP16", NUM_WARMUP, NUM_RUNS)
        if r: results.append(r)
    else:
        print(f"⚠️ Skipping OpenVINO FP16: {fp16_dir} not found")
    
    # 4. OpenVINO INT8 (auto-detects size)
    int8_dir = f"{MODELS_DIR}/best_int8_openvino_model"
    if os.path.exists(int8_dir):
        r = benchmark_openvino(int8_dir, "INT8", NUM_WARMUP, NUM_RUNS)
        if r: results.append(r)
    else:
        print(f"⚠️ Skipping OpenVINO INT8: {int8_dir} not found")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (416x416) - Intel CPU")
    print("=" * 70)
    print(f"{'Format':<20} {'Size (MB)':<12} {'Avg (ms)':<12} {'FPS':<10} {'Speedup':<10}")
    print("-" * 64)
    
    if results:
        baseline_ms = results[0]['avg_ms']
        for r in results:
            speedup = baseline_ms / r['avg_ms']
            print(f"{r['format']:<20} {r['size_mb']:<12.1f} {r['avg_ms']:<12.1f} {r['fps']:<10.1f} {speedup:.2f}x")
        
        best = max(results, key=lambda x: x['fps'])
        print(f"\n🏆 Best: {best['format']} at {best['fps']:.1f} FPS")
        
        # Save results
        with open("benchmarks/lisa_benchmark_results.txt", "w") as f:
            f.write("LISA Model Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'Format':<20} {'Size (MB)':<12} {'Avg (ms)':<12} {'FPS':<10}\n")
            f.write("-" * 54 + "\n")
            for r in results:
                f.write(f"{r['format']:<20} {r['size_mb']:<12.1f} {r['avg_ms']:<12.1f} {r['fps']:<10.1f}\n")
            f.write(f"\nBest: {best['format']} at {best['fps']:.1f} FPS\n")
        print("\n📄 Results saved to: benchmarks/lisa_benchmark_results.txt")
    else:
        print("No models found to benchmark!")


if __name__ == "__main__":
    main()
