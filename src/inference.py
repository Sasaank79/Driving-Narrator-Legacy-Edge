"""
Driving Narrator - ONNX Inference for Traffic Sign Detection
Optimized for CPU-only inference on Intel Mac (2016)

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --video path/to/video.mp4
    python inference.py --webcam
"""

import argparse
import time
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class TrafficSignDetector:
    """ONNX-based traffic sign detector optimized for CPU inference."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load ONNX model with CPU optimization
        print(f"Loading model: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on your CPU cores
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2]  # Usually 640
        
        # Class names (will be updated based on your dataset)
        # These are placeholders - update with your actual class names
        self.class_names = self._load_class_names()
        
        print(f"Model loaded successfully!")
        print(f"  Input size: {self.img_size}x{self.img_size}")
        print(f"  Classes: {len(self.class_names)}")
    
    def _load_class_names(self) -> list:
        """Load class names. Override this based on your dataset."""
        # Default traffic sign classes (update based on your Roboflow dataset)
        return [
            "stop", "yield", "speed_limit", "no_entry", "warning",
            "pedestrian_crossing", "traffic_light", "turn_left", 
            "turn_right", "straight", "no_parking", "one_way"
        ]
    
    def preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor and scaling info
        """
        original_h, original_w = image.shape[:2]
        
        # Calculate scaling to maintain aspect ratio
        scale = min(self.img_size / original_w, self.img_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (letterbox)
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert BGR to RGB, normalize, and transpose to NCHW
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        
        return blob, (scale, pad_w, pad_h, original_w, original_h)
    
    def postprocess(self, outputs: np.ndarray, scale_info: tuple) -> list:
        """
        Postprocess YOLO outputs to get detections.
        
        Args:
            outputs: Raw model outputs
            scale_info: Scaling information from preprocessing
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence, class_id, class_name), ...]
        """
        scale, pad_w, pad_h, orig_w, orig_h = scale_info
        
        # Output shape: (1, num_classes + 4, num_predictions)
        predictions = outputs[0].T  # Transpose to (num_predictions, num_classes + 4)
        
        # Split boxes and class scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        scores = predictions[:, 4:]  # Class scores
        
        # Get max class score and class id for each prediction
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Remove padding and scale back to original image
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Apply NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        indices = self._nms(boxes_for_nms, confidences, self.iou_threshold)
        
        # Build final detections
        detections = []
        for i in indices:
            class_name = self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else f"class_{class_ids[i]}"
            detections.append({
                'bbox': (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': class_name
            })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
        """Non-Maximum Suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray) -> list:
        """
        Run detection on an image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            List of detections
        """
        # Preprocess
        blob, scale_info = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})
        
        # Postprocess
        detections = self.postprocess(outputs[0], scale_info)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return output


def process_image(detector: TrafficSignDetector, image_path: str, output_dir: str):
    """Process a single image."""
    print(f"\nProcessing: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run detection
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = (time.time() - start_time) * 1000
    
    print(f"Inference time: {inference_time:.1f}ms")
    print(f"Detections: {len(detections)}")
    
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2%}")
    
    # Draw and save result
    output = detector.draw_detections(image, detections)
    output_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")


def process_video(detector: TrafficSignDetector, video_path: str, output_dir: str):
    """Process a video file."""
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    output_path = os.path.join(output_dir, f"result_{Path(video_path).name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    frame_times = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(frame)
        frame_time = (time.time() - start_time) * 1000
        frame_times.append(frame_time)
        
        # Draw results
        output_frame = detector.draw_detections(frame, detections)
        
        # Add FPS overlay
        avg_fps = 1000 / np.mean(frame_times[-30:]) if frame_times else 0
        cv2.putText(output_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({avg_fps:.1f} FPS)")
    
    cap.release()
    out.release()
    
    avg_time = np.mean(frame_times)
    print(f"\nComplete! Processed {frame_count} frames")
    print(f"Average inference time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    print(f"Saved: {output_path}")


def process_webcam(detector: TrafficSignDetector):
    """Process webcam feed in real-time."""
    print("\nStarting webcam... Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(frame)
        frame_time = (time.time() - start_time) * 1000
        frame_times.append(frame_time)
        
        # Draw results
        output = detector.draw_detections(frame, detections)
        
        # Add FPS overlay
        avg_fps = 1000 / np.mean(frame_times[-30:]) if frame_times else 0
        cv2.putText(output, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Traffic Sign Detection", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Average inference time: {np.mean(frame_times):.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Detection - ONNX Inference")
    parser.add_argument("--model", type=str, default="models/best.onnx", help="Path to ONNX model")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please download best.onnx from Google Drive and place it in the models/ folder")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = TrafficSignDetector(args.model, args.conf, args.iou)
    
    # Process input
    if args.image:
        process_image(detector, args.image, args.output)
    elif args.video:
        process_video(detector, args.video, args.output)
    elif args.webcam:
        process_webcam(detector)
    else:
        print("Please specify --image, --video, or --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
