"""
Domain Interfaces - Abstract base classes for dependency inversion
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from .entities import PredictionResult


class IPredictor(ABC):
    """Interface for prediction service."""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> PredictionResult:
        """Make prediction on an image."""
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the ML model."""
        pass


class IImageProcessor(ABC):
    """Interface for image processing service."""
    
    @abstractmethod
    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for prediction."""
        pass
    
    @abstractmethod
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract face from image."""
        pass
