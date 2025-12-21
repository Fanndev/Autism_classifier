"""
Image Processor - Handles image preprocessing with MTCNN face detection
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from mtcnn import MTCNN

from ..domain.interfaces import IImageProcessor


class MTCNNImageProcessor(IImageProcessor):
    """Image processor using MTCNN for face detection and alignment."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self._detector = None
    
    @property
    def detector(self) -> MTCNN:
        """Lazy loading of MTCNN detector."""
        if self._detector is None:
            self._detector = MTCNN()
        return self._detector
    
    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image: read, detect face, align, crop, and normalize.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array or None if face not detected
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect and process face
        return self.detect_face(img)
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face, align, crop, and resize.
        
        Args:
            image: RGB image array
            
        Returns:
            Processed face image or None if no face detected
        """
        # Detect faces
        detections = self.detector.detect_faces(image)
        if not detections:
            return None
        
        # Get the first (main) face
        detection = detections[0]
        keypoints = detection['keypoints']
        box = detection['box']
        
        # Align face based on eye positions
        aligned_image = self._align_face(image, keypoints)
        
        # Crop face with margin
        face_img = self._crop_face(aligned_image, box, margin=0.3)
        if face_img.size == 0:
            return None
        
        # Resize and normalize
        face_img = cv2.resize(face_img, self.target_size)
        face_img = face_img.astype(np.float32) / 255.0
        
        return face_img
    
    def _align_face(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Align face based on eye positions.
        
        Args:
            image: Input image
            keypoints: Dictionary with facial keypoints
            
        Returns:
            Aligned image
        """
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        return cv2.warpAffine(image, M, (w, h))
    
    def _crop_face(self, image: np.ndarray, box: list, margin: float = 0.3) -> np.ndarray:
        """
        Crop face region with margin.
        
        Args:
            image: Input image
            box: Bounding box [x, y, w, h]
            margin: Margin around the face
            
        Returns:
            Cropped face image
        """
        x, y, w, h = box
        
        # Calculate square crop with margin
        size = int(max(w, h) * (1 + margin))
        cx, cy = x + w // 2, y + h // 2
        
        # Ensure coordinates are within image bounds
        x1 = max(cx - size // 2, 0)
        y1 = max(cy - size // 2, 0)
        x2 = min(cx + size // 2, image.shape[1])
        y2 = min(cy + size // 2, image.shape[0])
        
        return image[y1:y2, x1:x2]
