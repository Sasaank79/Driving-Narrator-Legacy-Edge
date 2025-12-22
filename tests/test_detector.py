"""
Pytest tests for Driving Narrator
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import FPSCounter, resize_with_aspect_ratio, get_model_size


class TestFPSCounter:
    """Tests for FPS tracking utility."""
    
    def test_initial_fps_is_zero(self):
        """FPS should be 0 with no measurements."""
        counter = FPSCounter()
        assert counter.fps == 0.0
    
    def test_fps_after_single_update(self):
        """FPS should be 0 after single update (need at least 2)."""
        counter = FPSCounter()
        counter.update()
        assert counter.fps == 0.0
    
    def test_fps_positive_after_multiple_updates(self):
        """FPS should be positive after multiple updates."""
        import time
        counter = FPSCounter(window_size=10)
        for _ in range(5):
            counter.update()
            time.sleep(0.01)  # 10ms delay
        assert counter.fps > 0
    
    def test_window_size_limits_history(self):
        """Timestamps should be limited to window size."""
        counter = FPSCounter(window_size=5)
        for _ in range(10):
            counter.update()
        assert len(counter.timestamps) <= 5


class TestResizeWithAspectRatio:
    """Tests for image resizing utility."""
    
    def test_square_image(self):
        """Square image should resize without padding issues."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        resized, scale, padding = resize_with_aspect_ratio(img, 416)
        assert resized.shape == (416, 416, 3)
    
    def test_wide_image_padding(self):
        """Wide image should have top/bottom padding."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # 2:1 aspect
        resized, scale, padding = resize_with_aspect_ratio(img, 416)
        assert resized.shape == (416, 416, 3)
        assert padding[1] > 0  # y padding
    
    def test_tall_image_padding(self):
        """Tall image should have left/right padding."""
        img = np.zeros((200, 100, 3), dtype=np.uint8)  # 1:2 aspect
        resized, scale, padding = resize_with_aspect_ratio(img, 416)
        assert resized.shape == (416, 416, 3)
        assert padding[0] > 0  # x padding
    
    def test_scale_is_correct(self):
        """Scale factor should be correct."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized, scale, padding = resize_with_aspect_ratio(img, 400)
        assert scale == 2.0  # 400/200


class TestGetModelSize:
    """Tests for model size utility."""
    
    def test_nonexistent_path_returns_zero(self):
        """Non-existent path should return 0."""
        size = get_model_size("/nonexistent/path")
        assert size == 0.0
    
    def test_existing_file_returns_positive(self):
        """Existing file should return positive size."""
        # Use this test file itself
        size = get_model_size(__file__)
        assert size > 0


class TestDetectorIntegration:
    """Integration tests for TrafficSignDetector (requires model files)."""
    
    @pytest.fixture
    def model_path(self):
        """Path to INT8 model for testing."""
        return Path(__file__).parent.parent / "models" / "best_int8_openvino_model"
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "models" / "best_int8_openvino_model").exists(),
        reason="INT8 model not found"
    )
    def test_detector_loads(self, model_path):
        """Detector should load without errors."""
        from src.detector import TrafficSignDetector
        detector = TrafficSignDetector(model_path=str(model_path))
        assert detector.num_classes == 47
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "models" / "best_int8_openvino_model").exists(),
        reason="INT8 model not found"
    )
    def test_detector_inference(self, model_path):
        """Detector should run inference on dummy image."""
        from src.detector import TrafficSignDetector
        detector = TrafficSignDetector(model_path=str(model_path))
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_image)
        
        assert "boxes" in result
        assert "scores" in result
        assert "classes" in result
        assert "class_names" in result
