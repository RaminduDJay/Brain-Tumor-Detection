
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from collections import Counter
from config import Config

class BrainTumorDataLoader:
    """
    Data loader for brain tumor dataset with comprehensive analysis capabilities
    """
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = Path(dataset_path) if dataset_path else Config.DATASET_PATH
        self.training_path = self.dataset_path / "Training"
        self.testing_path = self.dataset_path / "Testing"
        self.class_names = Config.CLASS_NAMES
        
    def get_dataset_overview(self) -> Dict:
        """
        Get comprehensive dataset overview including file counts, sizes, etc.
        """
        overview = {
            'training': {},
            'testing': {},
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # Analyze training data
        for class_name in self.class_names:
            class_path = self.training_path / class_name
            if class_path.exists():
                files = list(class_path.glob("*.jpg"))
                overview['training'][class_name] = {
                    'count': len(files),
                    'files': [f.name for f in files[:5]]  # First 5 files
                }
        
        # Analyze testing data
        for class_name in self.class_names:
            class_path = self.testing_path / class_name
            if class_path.exists():
                files = list(class_path.glob("*.jpg"))
                overview['testing'][class_name] = {
                    'count': len(files),
                    'files': [f.name for f in files[:5]]  # First 5 files
                }
        
        return overview
    
    def load_image_paths_and_labels(self, split: str = 'training') -> Tuple[List[str], List[str]]:
        """
        Load all image paths and corresponding labels
        
        Args:
            split: 'training' or 'testing'
            
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        base_path = self.training_path if split == 'training' else self.testing_path
        
        for class_name in self.class_names:
            class_path = base_path / class_name
            if class_path.exists():
                for image_file in class_path.glob("*.jpg"):
                    image_paths.append(str(image_file))
                    labels.append(class_name)
        
        return image_paths, labels
    
    def load_sample_images(self, n_samples: int = 5) -> Dict:
        """
        Load sample images from each class for visualization
        
        Args:
            n_samples: Number of samples per class
            
        Returns:
            Dictionary with class names as keys and image arrays as values
        """
        samples = {}
        
        for class_name in self.class_names:
            class_path = self.training_path / class_name
            samples[class_name] = []
            
            if class_path.exists():
                image_files = list(class_path.glob("*.jpg"))[:n_samples]
                
                for image_file in image_files:
                    img = cv2.imread(str(image_file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    samples[class_name].append({
                        'image': img,
                        'filename': image_file.name,
                        'path': str(image_file)
                    })
        
        return samples
    
    def analyze_image_properties(self, split: str = 'training') -> pd.DataFrame:
        """
        Analyze image properties like dimensions, file sizes, etc.
        
        Args:
            split: 'training' or 'testing'
            
        Returns:
            DataFrame with image analysis results
        """
        image_paths, labels = self.load_image_paths_and_labels(split)
        
        analysis_data = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                # Load image
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Get image properties
                    height, width, channels = img.shape
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    
                    # Calculate basic statistics
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mean_intensity = np.mean(gray_img)
                    std_intensity = np.std(gray_img)
                    
                    analysis_data.append({
                        'filename': os.path.basename(img_path),
                        'class': label,
                        'width': width,
                        'height': height,
                        'channels': channels,
                        'file_size_kb': file_size,
                        'mean_intensity': mean_intensity,
                        'std_intensity': std_intensity,
                        'aspect_ratio': width / height
                    })
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return pd.DataFrame(analysis_data)