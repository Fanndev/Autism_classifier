"""
Predictor - ML model wrapper for autism classification
"""
import numpy as np
from typing import Optional
from pathlib import Path
import tensorflow as tf
from django.conf import settings

from ..domain.interfaces import IPredictor
from ..domain.entities import PredictionResult


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
