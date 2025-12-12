"""
Benchmark YOLO11n: ONNX vs OpenVINO FP32 vs OpenVINO INT8
All comparisons at 416x416 for Intel CPU.
"""

import time
import os
import numpy as np


def export_onnx(img_size: int = 416):
    """Export YOLO11n to ONNX."""
    from ultralytics import YOLO
    
    output_name = f"yolo11n_{img_size}.onnx"
    
    if os.path.exists(output_name):
        print(f"ONNX model exists: {output_name}")
        return output_name
    
    print(f"Exporting YOLO11n to ONNX ({img_size}x{img_size})...")
    model = YOLO('yolo11n.pt')
    onnx_path = model.export(format='onnx', imgsz=img_size, simplify=True, opset=12)
    os.rename(onnx_path, output_name)
    return output_name


def export_openvino_fp32(img_size: int = 416):
    """Export YOLO11n to OpenVINO FP32."""
    from ultralytics import YOLO
    
    xml_path = "yolo11n_openvino_model/yolo11n.xml"
    
    if os.path.exists(xml_path):
        print(f"OpenVINO FP32 model exists: {xml_path}")
        return xml_path
    
    print(f"Exporting YOLO11n to OpenVINO FP32 ({img_size}x{img_size})...")
    model = YOLO('yolo11n.pt')
    model.export(format='openvino', imgsz=img_size, half=False)
    return xml_path


def export_openvino_int8(img_size: int = 416):
    """Export YOLO11n to OpenVINO INT8."""
    from ultralytics import YOLO
    
    int8_dir = "yolo11n_openvino_int8"
    xml_path = f"{int8_dir}/yolo11n.xml"
    
    if os.path.exists(xml_path):
        print(f"OpenVINO INT8 model exists: {xml_path}")
        return xml_path
    
    print(f"Exporting YOLO11n to OpenVINO INT8 ({img_size}x{img_size})...")
    print("Note: This will download COCO128 for calibration...")
    
    model = YOLO('yolo11n.pt')
    model.export(format='openvino', imgsz=img_size, half=False, int8=True, data='coco128.yaml')
    
    # Rename output directory
    default_dir = "yolo11n_int8_openvino_model"
    if os.path.exists(default_dir) and not os.path.exists(int8_dir):
        os.rename(default_dir, int8_dir)
    
    return xml_path


def benchmark_onnx(model_path: str, img_size: int, num_warmup: int = 5, num_runs: int = 50):
    """Benchmark ONNX Runtime."""
    import onnxruntime as ort
    
    print("\n" + "=" * 60)
    print(f"ONNX Runtime ({img_size}x{img_size})")
    print("=" * 60)
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    
    # Warmup & benchmark
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})
    
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
    
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    print(f"Result: {avg_time:.1f}ms, {fps:.1f} FPS")
    
    return {'runtime': 'ONNX', 'avg_ms': avg_time, 'fps': fps, 'size_mb': size_mb}


def benchmark_openvino(xml_path: str, img_size: int, label: str, num_warmup: int = 5, num_runs: int = 50):
    """Benchmark OpenVINO."""
    from openvino import Core
    
    print("\n" + "=" * 60)
    print(f"OpenVINO {label} ({img_size}x{img_size})")
    print("=" * 60)
    
    bin_path = xml_path.replace('.xml', '.bin')
    size_mb = os.path.getsize(bin_path) / (1024 * 1024) if os.path.exists(bin_path) else 0
    print(f"Model size: {size_mb:.1f} MB")
    
    core = Core()
    model = core.read_model(xml_path)
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    input_tensor = compiled_model.input(0)
    
    dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    
    # Warmup & benchmark
    for _ in range(num_warmup):
        infer_request.infer({input_tensor: dummy_input})
    
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        infer_request.infer({input_tensor: dummy_input})
        times.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
    
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    print(f"Result: {avg_time:.1f}ms, {fps:.1f} FPS")
    
    return {'runtime': f'OpenVINO {label}', 'avg_ms': avg_time, 'fps': fps, 'size_mb': size_mb}


if __name__ == "__main__":
    IMG_SIZE = 416
    results = []
    
    # 1. ONNX Runtime baseline
    onnx_path = export_onnx(IMG_SIZE)
    results.append(benchmark_onnx(onnx_path, IMG_SIZE))
    
    # 2. OpenVINO FP32
    try:
        fp32_path = export_openvino_fp32(IMG_SIZE)
        results.append(benchmark_openvino(fp32_path, IMG_SIZE, "FP32"))
    except Exception as e:
        print(f"\n⚠️ OpenVINO FP32 failed: {e}")
    
    # 3. OpenVINO INT8
    try:
        int8_path = export_openvino_int8(IMG_SIZE)
        results.append(benchmark_openvino(int8_path, IMG_SIZE, "INT8"))
    except Exception as e:
        print(f"\n⚠️ OpenVINO INT8 failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FULL COMPARISON (416x416) - Intel CPU")
    print("=" * 70)
    print(f"{'Runtime':<20} {'Size (MB)':<12} {'Avg (ms)':<12} {'FPS':<10} {'Speedup':<10}")
    print("-" * 64)
    
    baseline_ms = results[0]['avg_ms']
    for r in results:
        speedup = baseline_ms / r['avg_ms']
        print(f"{r['runtime']:<20} {r['size_mb']:<12.1f} {r['avg_ms']:<12.1f} {r['fps']:<10.1f} {speedup:<10.1f}x")
    
    # Best result
    best = max(results, key=lambda x: x['fps'])
    print(f"\n🏆 Best: {best['runtime']} at {best['fps']:.1f} FPS")
