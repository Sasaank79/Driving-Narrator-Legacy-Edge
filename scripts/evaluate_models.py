"""
Evaluate INT8 model on test set
Compare with FP32 to measure quantization loss
"""
from ultralytics import YOLO
import json
from pathlib import Path

def evaluate_model(model_path, data_yaml, split='test', imgsz=640):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Split: {split}, Image size: {imgsz}")
    print(f"{'='*60}\n")
    
    if 'openvino' in str(model_path).lower():
        model = YOLO(model_path, task='detect')
    else:
        model = YOLO(model_path)
    
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        verbose=True
    )
    
    results = {
        'model': str(model_path),
        'mAP50': round(float(metrics.box.map50), 4),
        'mAP50_95': round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall': round(float(metrics.box.mr), 4),
    }
    
    print(f"\n--- Results ---")
    print(f"mAP@0.5:      {results['mAP50']:.2%}")
    print(f"mAP@0.5:0.95: {results['mAP50_95']:.2%}")
    print(f"Precision:    {results['precision']:.2%}")
    print(f"Recall:       {results['recall']:.2%}")
    
    return results

if __name__ == "__main__":
    # Dataset path
    data_yaml = 'LISA_stratified/data.yaml'
    
    # Models to evaluate (path, imgsz)
    # FP32 already completed: val=96.15%, test=96.86%
    models = {
        # 'FP32 (best.pt)': ('models/best.pt', 640),  # DONE
        'INT8 (OpenVINO)': ('models/best_int8_openvino_model/', 416),
    }
    
    # Splits to evaluate
    splits = ['val', 'test']
    
    all_results = {}
    
    for name, (path, imgsz) in models.items():
        if Path(path).exists():
            all_results[name] = {}
            for split in splits:
                result = evaluate_model(path, data_yaml, split, imgsz)
                all_results[name][split] = result
        else:
            print(f"⚠️ Not found: {path}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON: FP32 vs INT8")
    print("="*60)
    print(f"{'Model':<25} {'Split':<8} {'mAP@0.5':<12}")
    print("-"*60)
    
    for name, splits_data in all_results.items():
        for split, r in splits_data.items():
            print(f"{name:<25} {split:<8} {r['mAP50']:<12.2%}")
    
    # Save
    with open('reports/model_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved to: reports/model_comparison.json")
