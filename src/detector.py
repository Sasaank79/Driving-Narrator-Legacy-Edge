"""
Traffic Sign Detector - Core detection class wrapping YOLOv11
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")


class TrafficSignDetector:
    """
    Handles object detection requests using YOLOv11. 
    Design to run seamlessly across different model formats including PyTorch, ONNX, and OpenVINO 
    to ensure the best performance on available hardware.
    """
    
    # Default paths relative to project root
    MODELS = {
        "pytorch": "models/best.pt",
        "onnx": "models/best.onnx",
        "openvino_fp16": "models/best_openvino_model",
        "openvino_int8": "models/best_int8_openvino_model",
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "openvino_int8",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 416,
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to model file/directory. If None, uses default based on model_type.
            model_type: One of 'pytorch', 'onnx', 'openvino_fp16', 'openvino_int8'
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            img_size: Input image size for inference
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Resolve model path
        if model_path is None:
            if model_type not in self.MODELS:
                raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(self.MODELS.keys())}")
            model_path = self.MODELS[model_type]
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
    
    def detect(
        self,
        image: np.ndarray,
        return_annotated: bool = False,
    ) -> Dict[str, Any]:
        """
        Run detection on a single image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            return_annotated: If True, include annotated image in results
            
        Returns:
            Dictionary with 'boxes', 'scores', 'classes', and optionally 'annotated_image'
        """
        results = self.model(
            image,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]
        
        output = {
            "boxes": results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([]),
            "scores": results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.array([]),
            "classes": results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) > 0 else np.array([]),
            "class_names": [self.class_names[int(c)] for c in results.boxes.cls] if len(results.boxes) > 0 else [],
        }
        
        if return_annotated:
            output["annotated_image"] = results.plot()
        
        return output
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ):
        """
        Run detection on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show: Display video while processing
        """
        return self.model(
            video_path,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=output_path is not None,
            show=show,
        )
    
    def evaluate(self, data_yaml: str) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_yaml: Path to YOLO format data.yaml
            
        Returns:
            Dictionary with mAP@0.5, mAP@0.5:0.95, precision, recall
        """
        metrics = self.model.val(data=data_yaml, imgsz=self.img_size, verbose=False)
        return {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)
    
    def __repr__(self) -> str:
        return f"TrafficSignDetector(model={self.model_path.name}, classes={self.num_classes})"
