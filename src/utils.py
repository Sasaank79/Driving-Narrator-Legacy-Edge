"""
Utility functions for Driving Narrator
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from threading import Thread
import queue


class VideoReader:
    """
    Multi-threaded video reader for non-blocking frame capture.
    Uses producer-consumer pattern with queue.
    """
    
    def __init__(self, video_path: str, queue_size: int = 5):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def start(self) -> 'VideoReader':
        """Start background frame reading thread."""
        self.thread = Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        return self
    
    def _read_frames(self):
        """Background thread: continuously read frames into queue."""
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)  # Avoid busy wait
    
    def read(self) -> Optional[np.ndarray]:
        """Get next frame from queue (non-blocking)."""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the reader and release resources."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()


class FPSCounter:
    """Track and smooth FPS measurements."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: List[float] = []
    
    def update(self) -> float:
        """Record current timestamp and return current FPS."""
        now = time.time()
        self.timestamps.append(now)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
        
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
    
    @property
    def fps(self) -> float:
        """Get current smoothed FPS."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image maintaining aspect ratio with padding.
    
    Args:
        image: Input image (BGR)
        target_size: Target size (square)
        pad_color: Color for padding
        
    Returns:
        Tuple of (resized_image, scale, padding)
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    
    # Center the resized image
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return padded, scale, (pad_x, pad_y)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes on image.
    
    Args:
        image: Input image (BGR)
        boxes: Bounding boxes [[x1, y1, x2, y2], ...]
        labels: Class labels
        scores: Confidence scores
        color: Box color
        thickness: Line thickness
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        text = f"{label}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(result, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result


def get_model_size(model_path: str) -> float:
    """Get model size in MB."""
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total / (1024 * 1024)
    return 0.0
