"""
Driving Narrator - Traffic Sign Detection on Legacy Hardware
"""

__version__ = "3.0.0"
__author__ = "Surya Sasaank Yanamandra"

from .detector import TrafficSignDetector
from .utils import VideoReader, FPSCounter, draw_detections, get_model_size

__all__ = [
    "TrafficSignDetector",
    "VideoReader",
    "FPSCounter",
    "draw_detections",
    "get_model_size",
]
