import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
import logging

class MRIImagePreprocessor:
    """
    Comprehensive MRI image preprocessing for brain tumor detection
    
    This class handles all image preprocessing steps including:
    - Image loading and validation
    - Resizing and normalization
    - Noise reduction and enhancement
    - Intensity standardization
    - Quality control
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize_method: str = 'minmax',
                 enhance_contrast: bool = True,
                 reduce_noise: bool = True):
        """
        Initialize the preprocessor with configuration parameters
        
        Args:
            target_size: Target image dimensions (width, height)
            normalize_method: 'minmax', 'zscore', or 'histogram'
            enhance_contrast: Whether to apply contrast enhancement
            reduce_noise: Whether to apply noise reduction
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.enhance_contrast = enhance_contrast
        self.reduce_noise = reduce_noise
        
        # Initialize scalers
        self.scaler = MinMaxScaler() if normalize_method == 'minmax' else StandardScaler()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def load_and_validate_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image and perform basic validation
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image array or None if invalid
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic validation
            if image.shape[0] < 50 or image.shape[1] < 50:
                self.logger.warning(f"Image too small: {image_path}")
                return None
                
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale with optimal weights for medical imaging
        
        Args:
            image: RGB image array
            
        Returns:
            Grayscale image array
        """
        if len(image.shape) == 3:
            # Use weighted conversion optimized for medical imaging
            # Standard weights: R=0.299, G=0.587, B=0.114
            # Medical imaging optimized: R=0.2989, G=0.5870, B=0.1141
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1141])
            return gray.astype(np.uint8)
        return image
    
    def resize_image(self, image: np.ndarray, preserve_aspect: bool = False) -> np.ndarray:
        """
        Resize image to target dimensions
        
        Args:
            image: Input image array
            preserve_aspect: Whether to preserve aspect ratio with padding
            
        Returns:
            Resized image array
        """
        if preserve_aspect:
            # Calculate padding to preserve aspect ratio
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            # Calculate scale factor
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create padded image
            if len(image.shape) == 3:
                padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            else:
                padded = np.zeros((target_h, target_w), dtype=image.dtype)
            
            # Calculate padding offsets
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        else:
            # Simple resize without preserving aspect ratio
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques
        
        Args:
            image: Input image array
            
        Returns:
            Denoised image array
        """
        if not self.reduce_noise:
            return image
            
        # Apply bilateral filter for edge-preserving smoothing
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
        # Optional: Apply median filter for salt-and-pepper noise
        # denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using multiple techniques
        
        Args:
            image: Input image array
            
        Returns:
            Contrast-enhanced image array
        """
        if not self.enhance_contrast:
            return image
            
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images, apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
            
        return enhanced
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity values
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        if self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            normalized = (image - image.mean()) / (image.std() + 1e-8)
        elif self.normalize_method == 'histogram':
            # Histogram equalization
            if len(image.shape) == 3:
                # For color images, equalize each channel
                normalized = np.zeros_like(image, dtype=np.float32)
                for i in range(image.shape[2]):
                    normalized[:,:,i] = cv2.equalizeHist(image[:,:,i].astype(np.uint8)) / 255.0
            else:
                normalized = cv2.equalizeHist(image.astype(np.uint8)) / 255.0
        else:
            # Default: simple normalization to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            
        return normalized
    
    def preprocess_single_image(self, 
                              image_path: str, 
                              return_original: bool = False) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for a single image
        
        Args:
            image_path: Path to the image file
            return_original: Whether to return original image for comparison
            
        Returns:
            Dictionary containing processed image and metadata
        """
        result = {
            'success': False,
            'original_path': image_path,
            'processed_image': None,
            'original_image': None,
            'metadata': {}
        }
        
        try:
            # Load and validate image
            original_image = self.load_and_validate_image(image_path)
            if original_image is None:
                return result
                
            # Store original if requested
            if return_original:
                result['original_image'] = original_image.copy()
                
            # Get original dimensions
            original_shape = original_image.shape
            
            # Convert to grayscale
            gray_image = self.convert_to_grayscale(original_image)
            
            # Apply noise reduction
            denoised_image = self.reduce_noise(gray_image)
            
            # Enhance contrast
            enhanced_image = self.enhance_contrast(denoised_image)
            
            # Resize image
            resized_image = self.resize_image(enhanced_image, preserve_aspect=True)
            
            # Normalize intensity
            normalized_image = self.normalize_intensity(resized_image)
            
            # Store results
            result.update({
                'success': True,
                'processed_image': normalized_image,
                'metadata': {
                    'original_shape': original_shape,
                    'processed_shape': normalized_image.shape,
                    'normalization_method': self.normalize_method,
                    'target_size': self.target_size,
                    'intensity_range': (normalized_image.min(), normalized_image.max()),
                    'mean_intensity': normalized_image.mean(),
                    'std_intensity': normalized_image.std()
                }
            })
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            
        return result