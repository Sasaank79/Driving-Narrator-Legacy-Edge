"""
Benchmark YOLO11n ONNX with multiple input sizes.
Tests 640, 480, 416, and 320 for performance comparison.
"""

import time
import os
import numpy as np

def export_to_onnx(img_size: int = 640):
    """Export stock YOLO11n to ONNX format with specified input size."""
    from ultralytics import YOLO
    
    output_name = f"yolo11n_{img_size}.onnx"
    
    if os.path.exists(output_name):
        print(f"ONNX model already exists: {output_name}")
        return output_name
    
    print(f"Exporting YOLO11n to ONNX (input size: {img_size}x{img_size})...")
    model = YOLO('yolo11n.pt')
    
    onnx_path = model.export(
        format='onnx',
        imgsz=img_size,
        simplify=True,
        opset=12,
        dynamic=False,
        half=False
    )
    
    # Rename to include size
    os.rename(onnx_path, output_name)
    print(f"Exported to: {output_name}")
    return output_name


def benchmark_onnx(model_path: str, img_size: int, num_warmup: int = 5, num_runs: int = 50):
    """Benchmark ONNX model on CPU."""
    import onnxruntime as ort
    
    print("\n" + "=" * 60)
    print(f"YOLO11n ONNX Benchmark - {img_size}x{img_size}")
    print("=" * 60)
    
    # Load ONNX model
    print(f"\nLoading: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    
    # Warmup
    print(f"\nWarmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy_input})
    
    # Timed runs
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
    
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time
    
    print("\n" + "-" * 40)
    print(f"Results ({img_size}x{img_size}):")
    print(f"  Average: {avg_time:.1f}ms ± {std_time:.1f}ms")
    print(f"  Min/Max: {min_time:.1f}ms / {max_time:.1f}ms")
    print(f"  FPS:     {fps:.1f}")
    print("-" * 40)
    
    return {'img_size': img_size, 'avg_ms': avg_time, 'fps': fps, 'min_ms': min_time}


if __name__ == "__main__":
    results = []
    
    # Test all sizes
    sizes = [640, 480, 416, 320]
    
    for size in sizes:
        onnx_path = export_to_onnx(size)
        result = benchmark_onnx(onnx_path, size)
        results.append(result)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Size':<12} {'Avg (ms)':<12} {'FPS':<10} {'Best (ms)':<12} {'Real-time?':<10}")
    print("-" * 56)
    
    baseline_ms = results[0]['avg_ms']
    for r in results:
        speedup = baseline_ms / r['avg_ms']
        real_time = "✅ Yes" if r['fps'] >= 15 else ("⚠️ Near" if r['fps'] >= 10 else "❌ No")
        print(f"{r['img_size']}x{r['img_size']:<6} {r['avg_ms']:<12.1f} {r['fps']:<10.1f} {r['min_ms']:<12.1f} {real_time}")
    
    print("\n📈 Speedup vs 640x640:")
    for r in results[1:]:
        speedup = baseline_ms / r['avg_ms']
        print(f"   {r['img_size']}x{r['img_size']}: {speedup:.1f}x faster")
