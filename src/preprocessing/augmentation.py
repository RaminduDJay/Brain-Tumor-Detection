import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import random
from sklearn.utils import shuffle
import tensorflow as tf

class MedicalImageAugmentation:
    """
    Medical image augmentation specifically designed for brain MRI scans
    
    Ensures augmentations are medically appropriate and preserve diagnostic information
    """
    
    def __init__(self, 
                 rotation_range: float = 15,
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 flip_horizontal: bool = True,
                 flip_vertical: bool = False,
                 add_noise: bool = True,
                 elastic_deformation: bool = False):
        """
        Initialize augmentation parameters
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            zoom_range: Range for random zoom (min_zoom, max_zoom)
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            flip_horizontal: Whether to apply horizontal flipping
            flip_vertical: Whether to apply vertical flipping (usually False for brain scans)
            add_noise: Whether to add Gaussian noise
            elastic_deformation: Whether to apply elastic deformation
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.add_noise = add_noise
        self.elastic_deformation = elastic_deformation
        
    def rotate_image(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate image by specified or random angle
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees (random if None)
            
        Returns:
            Rotated image
        """
        if angle is None:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def zoom_image(self, image: np.ndarray, zoom_factor: Optional[float] = None) -> np.ndarray:
        """
        Apply zoom transformation to image
        
        Args:
            image: Input image array
            zoom_factor: Zoom factor (random if None)
            
        Returns:
            Zoomed image
        """
        if zoom_factor is None:
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
        
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if zoom_factor > 1.0:
            # Crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            zoomed = resized[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad to center
            if len(image.shape) == 3:
                zoomed = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
            else:
                zoomed = np.zeros((h, w), dtype=image.dtype)
            
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            zoomed[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return zoomed
    
    def adjust_brightness_contrast(self, 
                                 image: np.ndarray, 
                                 brightness_factor: Optional[float] = None,
                                 contrast_factor: Optional[float] = None) -> np.ndarray:
        """
        Adjust image brightness and contrast
        
        Args:
            image: Input image array
            brightness_factor: Brightness multiplication factor
            contrast_factor: Contrast multiplication factor
            
        Returns:
            Adjusted image
        """
        if brightness_factor is None:
            brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        if contrast_factor is None:
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Apply brightness and contrast adjustment
        # Formula: new_image = contrast * image + brightness_offset
        mean_intensity = np.mean(image)
        brightness_offset = (brightness_factor - 1.0) * mean_intensity
        
        adjusted = contrast_factor * image + brightness_offset
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 1 if image.max() <= 1 else 255)
        
        return adjusted.astype(image.dtype)
    
    def flip_image(self, image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """
        Flip image horizontally or vertically
        
        Args:
            image: Input image array
            horizontal: True for horizontal flip, False for vertical
            
        Returns:
            Flipped image
        """
        if horizontal:
            return cv2.flip(image, 1)  # Horizontal flip
        else:
            return cv2.flip(image, 0)  # Vertical flip
    
    def add_gaussian_noise(self, 
                          image: np.ndarray, 
                          noise_intensity: float = 0.02) -> np.ndarray:
        """
        Add Gaussian noise to image
        
        Args:
            image: Input image array
            noise_intensity: Standard deviation of noise
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_intensity, image.shape)
        noisy_image = image + noise
        
        # Clip values to valid range
        noisy_image = np.clip(noisy_image, 0, 1 if image.max() <= 1 else 255)
        
        return noisy_image.astype(image.dtype)
    
    def apply_elastic_deformation(self, 
                                image: np.ndarray,
                                alpha: float = 100,
                                sigma: float = 10) -> np.ndarray:
        """
        Apply elastic deformation to image
        
        Args:
            image: Input image array
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            
        Returns:
            Deformed image
        """
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)) * alpha
        dy = np.random.uniform(-1, 1, (h, w)) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply deformation
        deformed = cv2.remap(image, map_x, map_y, 
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        
        return deformed
    
    def augment_single_image(self, 
                           image: np.ndarray, 
                           augmentation_probability: float = 0.5) -> np.ndarray:
        """
        Apply random augmentations to a single image
        
        Args:
            image: Input image array
            augmentation_probability: Probability of applying each augmentation
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Apply rotations
        if random.random() < augmentation_probability:
            augmented = self.rotate_image(augmented)
        
        # Apply zoom
        if random.random() < augmentation_probability:
            augmented = self.zoom_image(augmented)
        
        # Apply brightness/contrast adjustment
        if random.random() < augmentation_probability:
            augmented = self.adjust_brightness_contrast(augmented)
        
        # Apply horizontal flip
        if self.flip_horizontal and random.random() < augmentation_probability:
            augmented = self.flip_image(augmented, horizontal=True)
        
        # Apply vertical flip (rarely used for brain scans)
        if self.flip_vertical and random.random() < augmentation_probability:
            augmented = self.flip_image(augmented, horizontal=False)
        
        # Add noise
        if self.add_noise and random.random() < augmentation_probability:
            augmented = self.add_gaussian_noise(augmented)
        
        # Apply elastic deformation
        if self.elastic_deformation and random.random() < augmentation_probability:
            augmented = self.apply_elastic_deformation(augmented)
        
        return augmented
    
    def create_augmented_dataset(self, 
                               images: List[np.ndarray], 
                               labels: List[str],
                               augmentation_factor: int = 2) -> Tuple[List[np.ndarray], List[str]]:
        """
        Create augmented dataset from original images
        
        Args:
            images: List of original images
            labels: List of corresponding labels
            augmentation_factor: Number of augmented versions per original image
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        # Add original images
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        
        # Generate augmented versions
        for _ in range(augmentation_factor):
            for img, label in zip(images, labels):
                augmented_img = self.augment_single_image(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)
        
        # Shuffle the dataset
        augmented_images, augmented_labels = shuffle(augmented_images, augmented_labels)
        
        return augmented_images, augmented_labels