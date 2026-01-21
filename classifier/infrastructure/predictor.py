"""
Predictor - ML model wrapper for autism classification
"""
import numpy as np
from typing import Optional
from pathlib import Path
import tensorflow as tf
import torch
import torch.nn as nn
from django.conf import settings

from ..domain.interfaces import IPredictor
from ..domain.entities import PredictionResult


# ============== PyTorch Model Architecture ==============
class AutismCNN(nn.Module):
    """PyTorch CNN model for autism classification (same as training notebook)."""
    
    def __init__(self):
        super(AutismCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        # Dense Layers (224/16 = 14)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# ============== PyTorch Predictor ==============
class PyTorchPredictor(IPredictor):
    """Predictor using PyTorch CNN model."""
    
    CLASSES = ['Autistic', 'Non_Autistic']
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_DIR / 'autism_cnn_model_by_qullah.pth'
        self._model = None
        self._is_loaded = False
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def model(self):
        """Lazy loading of the ML model."""
        if self._model is None:
            self.load_model()
        return self._model
    
    def load_model(self) -> bool:
        """
        Load the trained PyTorch model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self._model = AutismCNN()
            self._model.load_state_dict(torch.load(str(self.model_path), map_location=self._device, weights_only=True))
            self._model.to(self._device)
            self._model.eval()
            self._is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            self._is_loaded = False
            return False
    
    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Make prediction on preprocessed image.
        
        Args:
            image: Preprocessed image array (224, 224, 3) in HWC format
            
        Returns:
            PredictionResult with label and confidence
        """
        # Convert image to PyTorch format: (N, C, H, W)
        if len(image.shape) == 3:
            # HWC -> CHW -> NCHW
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:
            # Already batched NHWC -> NCHW
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2)
        
        image_tensor = image_tensor.to(self._device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(image_tensor).item()
        
        # PyTorch model: < 0.5 = Autistic (class 0), >= 0.5 = Non_Autistic (class 1)
        is_autistic = prediction < 0.5
        
        if is_autistic:
            label = self.CLASSES[0]  # Autistic
            confidence = 1 - prediction  # Confidence for Autistic
        else:
            label = self.CLASSES[1]  # Non_Autistic
            confidence = prediction  # Confidence for Non_Autistic
        
        return PredictionResult(
            label=label,
            confidence=float(confidence),
            is_autistic=is_autistic,
            processed_image=image
        )


class AutismPredictor(IPredictor):
    """Predictor using the trained CNN model."""
    
    CLASSES = ['Autistic', 'Non_Autistic']
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self._model = None
        self._is_loaded = False
    
    @property
    def model(self):
        """Lazy loading of the ML model."""
        if self._model is None:
            self.load_model()
        return self._model
    
    def load_model(self) -> bool:
        """
        Load the trained Keras model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Suppress TensorFlow logging
            tf.get_logger().setLevel('ERROR')
            
            self._model = tf.keras.models.load_model(str(self.model_path))
            self._is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self._is_loaded = False
            return False
    
    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Make prediction on preprocessed image.
        
        Args:
            image: Preprocessed image array (224, 224, 3)
            
        Returns:
            PredictionResult with label and confidence
        """
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get prediction
        prediction = self.model.predict(image, verbose=0)[0][0]
        
        # Model output: >= 0.5 = Autistic, < 0.5 = Non_Autistic
        is_autistic = prediction >= 0.5
        
        if is_autistic:
            label = self.CLASSES[0]  # Autistic
            confidence = prediction  # Confidence for Autistic
        else:
            label = self.CLASSES[1]  # Non_Autistic
            confidence = 1 - prediction  # Confidence for Non_Autistic
        
        return PredictionResult(
            label=label,
            confidence=float(confidence),
            is_autistic=is_autistic,
            processed_image=image[0] if len(image.shape) == 4 else image
        )
