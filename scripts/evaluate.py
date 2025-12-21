#!/usr/bin/env python3
"""
Evaluate trained model on test set and generate per-class metrics.

Usage:
    python scripts/evaluate.py --model models/best_int8_openvino_model --data path/to/data.yaml
    python scripts/evaluate.py --model models/best.pt --output results/
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Please install ultralytics: pip install ultralytics")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate traffic sign detection model on test set"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_int8_openvino_model",
        help="Path to model file or directory",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml file (YOLO format)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="Inference image size (default: 416)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data config not found: {data_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Evaluating on: {data_path}")
    print(f"Image size: {args.img_size}, Confidence: {args.conf}")
    
    # Run evaluation
    metrics = model.val(
        data=str(data_path),
        imgsz=args.img_size,
        conf=args.conf,
        verbose=args.verbose,
        save_json=True,
        plots=True,
    )
    
    # Extract results
    results = {
        "model": str(model_path),
        "data": str(data_path),
        "img_size": args.img_size,
        "conf_threshold": args.conf,
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": {},
    }
    
    # Per-class metrics
    class_names = model.names
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for i, ap in enumerate(metrics.box.ap50):
            if i < len(class_names):
                results["per_class"][class_names[i]] = {
                    "ap50": float(ap),
                }
    
    # Save results
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP@0.5:      {results['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results['metrics']['mAP50_95']:.4f}")
    print(f"Precision:    {results['metrics']['precision']:.4f}")
    print(f"Recall:       {results['metrics']['recall']:.4f}")
    print("=" * 50)
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
