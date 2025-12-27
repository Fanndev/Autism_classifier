"""
Application Services - Use cases and business logic orchestration
"""
from typing import Optional, Tuple
from dataclasses import dataclass

from ..domain.entities import PredictionResult, ImageData
from ..domain.interfaces import IPredictor, IImageProcessor
from ..infrastructure.predictor import AutismPredictor
from ..infrastructure.image_processor import MTCNNImageProcessor


@dataclass
class ClassificationResult:
    """Result of classification service."""
    success: bool
    prediction: Optional[PredictionResult] = None
    error_message: Optional[str] = None


class AutismClassifierService:
    """
    Main service for autism classification.
    Orchestrates image processing and prediction.
    """
    
    def __init__(
        self,
        predictor: Optional[IPredictor] = None,
        image_processor: Optional[IImageProcessor] = None
    ):
        self._predictor = predictor
        self._image_processor = image_processor
    
    @property
    def predictor(self) -> IPredictor:
        """Lazy initialization of predictor."""
        if self._predictor is None:
            self._predictor = AutismPredictor()
        return self._predictor
    
    @property
    def image_processor(self) -> IImageProcessor:
        """Lazy initialization of image processor."""
        if self._image_processor is None:
            self._image_processor = MTCNNImageProcessor()
        return self._image_processor
    
    def classify_image(self, image_path: str) -> ClassificationResult:
        """
        Classify an image for autism detection.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ClassificationResult with prediction or error message
        """
        try:
            # Preprocess image
            processed_image = self.image_processor.preprocess(image_path)
            
            if processed_image is None:
                return ClassificationResult(
                    success=False,
                    error_message="Tidak dapat mendeteksi wajah pada gambar. "
                                  "Pastikan gambar berisi wajah yang jelas dan terlihat."
                )
            
            # Make prediction
            prediction = self.predictor.predict(processed_image)
            
            return ClassificationResult(
                success=True,
                prediction=prediction
            )
            
        except Exception as e:
            return ClassificationResult(
                success=False,
                error_message=f"Terjadi kesalahan saat memproses gambar: {str(e)}"
            )
    
    def validate_image(self, image_data: ImageData) -> Tuple[bool, str]:
        """
        Validate uploaded image.
        
        Args:
            image_data: ImageData entity
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not image_data.is_valid_image():
            return False, "Format file tidak didukung. Gunakan JPG atau PNG."
        
        return True, ""


# Service factory function
def get_classifier_service(model_filename: Optional[str] = None) -> AutismClassifierService:
    """
    Create classifier service instance with specified model.
    
    Args:
        model_filename: Name of the model file to use
        
    Returns:
        AutismClassifierService instance
    """
    model_path = None
    if model_filename:
        model_path = settings.MODEL_DIR / model_filename
    
    return AutismClassifierService(model_path=model_path)
