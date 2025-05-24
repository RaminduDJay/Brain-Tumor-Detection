import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class BrainRegionSegmentation:
    """
    K-means based brain region segmentation for feature extraction
    
    This class performs brain tissue segmentation using K-means clustering
    to identify different brain regions and potential tumor areas.
    """
    
    def __init__(self, 
                 n_clusters: int = 4,
                 random_state: int = 42,
                 preprocessing_steps: List[str] = None):
        """
        Initialize brain segmentation with K-means parameters
        
        Args:
            n_clusters: Number of clusters (typically 4: CSF, Gray Matter, White Matter, Tumor)
            random_state: Random seed for reproducibility
            preprocessing_steps: List of preprocessing steps to apply
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.preprocessing_steps = preprocessing_steps or ['blur', 'enhance']
        
        # Initialize K-means
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        # Cluster interpretation (typical medical imaging)
        self.cluster_names = {
            0: 'CSF',           # Cerebrospinal Fluid (darkest)
            1: 'Gray Matter',   # Gray matter
            2: 'White Matter',  # White matter  
            3: 'Abnormal/Tumor' # Abnormal tissue/tumor (brightest)
        }
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing specific to segmentation
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image ready for segmentation
        """
        processed = image.copy()
        
        if 'blur' in self.preprocessing_steps:
            # Apply Gaussian blur to reduce noise
            processed = cv2.GaussianBlur(processed, (5, 5), 0)
            
        if 'enhance' in self.preprocessing_steps:
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply((processed * 255).astype(np.uint8)) / 255.0
            
        if 'morph' in self.preprocessing_steps:
            # Apply morphological operations
            kernel = np.ones((3,3), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
        return processed
    
    def extract_features_for_clustering(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features for K-means clustering
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Feature array for clustering
        """
        features = []
        
        # 1. Pixel intensities (primary feature)
        intensities = image.flatten().reshape(-1, 1)
        features.append(intensities)
        
        # 2. Local texture features (optional)
        # Calculate local standard deviation
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(image, -1, kernel)
        local_variance = cv2.filter2D(image**2, -1, kernel) - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        
        texture_features = local_std.flatten().reshape(-1, 1)
        features.append(texture_features)
        
        # 3. Spatial coordinates (optional - helps with spatial consistency)
        h, w = image.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Normalize coordinates
        x_norm = x_coords.flatten().reshape(-1, 1) / w
        y_norm = y_coords.flatten().reshape(-1, 1) / h
        
        features.extend([x_norm, y_norm])
        
        # Combine all features
        combined_features = np.hstack(features)
        
        # Standardize features
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(combined_features)
        
        return standardized_features
    
    def perform_segmentation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform K-means brain segmentation
        
        Args:
            image: Input grayscale image (normalized to [0,1])
            
        Returns:
            Dictionary containing segmentation results
        """
        # Preprocess image for segmentation
        processed_img = self.preprocess_for_segmentation(image)
        
        # Extract features for clustering
        features = self.extract_features_for_clustering(processed_img)
        
        # Perform K-means clustering
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Reshape labels back to image dimensions
        segmented_image = cluster_labels.reshape(image.shape)
        
        # Sort clusters by intensity (darkest to brightest)
        cluster_intensities = []
        for i in range(self.n_clusters):
            mask = segmented_image == i
            if np.any(mask):
                cluster_intensities.append((i, np.mean(image[mask])))
        
        # Sort by intensity
        cluster_intensities.sort(key=lambda x: x[1])
        
        # Create intensity-ordered segmentation
        ordered_segmentation = np.zeros_like(segmented_image)
        for new_label, (old_label, _) in enumerate(cluster_intensities):
            ordered_segmentation[segmented_image == old_label] = new_label
        
        # Create individual masks for each region
        region_masks = {}
        for i in range(self.n_clusters):
            region_masks[self.cluster_names[i]] = (ordered_segmentation == i)
        
        return {
            'segmented_image': ordered_segmentation,
            'region_masks': region_masks,
            'cluster_centers': self.kmeans.cluster_centers_,
            'n_clusters': self.n_clusters,
            'original_image': image
        }
    
    def extract_region_features(self, 
                              image: np.ndarray, 
                              segmentation_result: Dict) -> Dict[str, Dict]:
        """
        Extract statistical features from each segmented region
        
        Args:
            image: Original grayscale image
            segmentation_result: Result from perform_segmentation()
            
        Returns:
            Dictionary of features for each brain region
        """
        region_features = {}
        
        for region_name, mask in segmentation_result['region_masks'].items():
            if np.any(mask):
                region_pixels = image[mask]
                
                features = {
                    # Basic statistics
                    'mean_intensity': np.mean(region_pixels),
                    'std_intensity': np.std(region_pixels),
                    'min_intensity': np.min(region_pixels),
                    'max_intensity': np.max(region_pixels),
                    'median_intensity': np.median(region_pixels),
                    
                    # Shape and size features
                    'area': np.sum(mask),
                    'area_ratio': np.sum(mask) / mask.size,
                    
                    # Distribution features
                    'skewness': self._calculate_skewness(region_pixels),
                    'kurtosis': self._calculate_kurtosis(region_pixels),
                    
                    # Texture features
                    'entropy': self._calculate_entropy(region_pixels),
                    'contrast': np.std(region_pixels) / (np.mean(region_pixels) + 1e-8)
                }
                
                region_features[region_name] = features
            else:
                # Empty region
                region_features[region_name] = {key: 0.0 for key in [
                    'mean_intensity', 'std_intensity', 'min_intensity', 
                    'max_intensity', 'median_intensity', 'area', 'area_ratio',
                    'skewness', 'kurtosis', 'entropy', 'contrast'
                ]}
        
        return region_features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data: np.ndarray, bins: int = 256) -> float:
        """Calculate entropy of intensity distribution"""
        hist, _ = np.histogram(data, bins=bins, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize to probabilities
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0