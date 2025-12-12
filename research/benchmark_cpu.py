"""
Benchmark stock YOLO11 Nano on local CPU.
Tests inference speed using the pretrained model.
"""

import time
import numpy as np

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    from ultralytics import YOLO


def benchmark_yolo11n(num_warmup: int = 5, num_runs: int = 50):
    """
    Benchmark YOLO11n on CPU.
    
    Args:
        num_warmup: Number of warmup runs (not counted)
        num_runs: Number of timed runs
    """
    print("=" * 60)
    print("YOLO11n CPU Benchmark - Stock Pretrained Model")
    print("=" * 60)
    
    # Load pretrained YOLO11n (downloads if not cached)
    print("\nLoading YOLO11n pretrained model...")
    model = YOLO('yolo11n.pt')
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Create dummy image (640x640 RGB)
    print(f"\nCreating test image (640x640)...")
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup runs
    print(f"\nWarmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = model(dummy_image, verbose=False)
    
    # Timed runs
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_image, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 10 == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.1f}ms")
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Average inference time: {avg_time:.1f}ms ± {std_time:.1f}ms")
    print(f"Min / Max time:         {min_time:.1f}ms / {max_time:.1f}ms")
    print(f"Average FPS:            {fps:.1f}")
    print("=" * 60)
    
    # Performance assessment for Intel Mac 2016
    print("\n📊 Performance Assessment:")
    if fps >= 15:
        print("   ✅ Real-time capable (≥15 FPS)")
    elif fps >= 10:
        print("   ⚠️ Near real-time (10-15 FPS)")
    elif fps >= 5:
        print("   ⚠️ Slow but usable (5-10 FPS)")
    else:
        print("   ❌ Too slow for real-time (<5 FPS)")
    
    return {
        'avg_ms': avg_time,
        'std_ms': std_time,
        'fps': fps,
        'min_ms': min_time,
        'max_ms': max_time
    }


if __name__ == "__main__":
    results = benchmark_yolo11n()
