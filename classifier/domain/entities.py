"""
Domain Entities - Core business objects
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PredictionResult:
    """Entity representing a prediction result."""
    label: str
    confidence: float
    is_autistic: bool
    processed_image: Optional[np.ndarray] = None
    
    @property
    def confidence_percentage(self) -> str:
        """Return confidence as formatted percentage."""
        return f"{self.confidence * 100:.1f}%"
    
    @property
    def result_message(self) -> str:
        """Return human-readable result message."""
        if self.is_autistic:
            return f"Terdeteksi kemungkinan Autistic dengan keyakinan {self.confidence_percentage}"
        return f"Terdeteksi Non-Autistic dengan keyakinan {self.confidence_percentage}"


@dataclass
class ImageData:
    """Entity representing uploaded image data."""
    file_path: str
    original_filename: str
    content_type: str
    
    def is_valid_image(self) -> bool:
        """Check if the file is a valid image type."""
        valid_types = ['image/jpeg', 'image/png', 'image/jpg']
        return self.content_type.lower() in valid_types
