import numpy as np
import cv2
from typing import Dict, List, Tuple
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage
from scipy.stats import skew, kurtosis
import pandas as pd

class ComprehensiveFeatureExtractor:
    """
    Extract comprehensive features from brain MRI images for traditional ML models
    
    Features include:
    - Statistical features
    - Texture features (GLCM, LBP)
    - Shape features from segmented regions
    - Intensity distribution features
    - Gradient-based features
    """
    
    def __init__(self):
        # GLCM parameters
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, 45, 90, 135]
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        
    def extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from the image
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(image)
        features['std'] = np.std(image)
        features['var'] = np.var(image)
        features['min'] = np.min(image)
        features['max'] = np.max(image)
        features['median'] = np.median(image)
        features['range'] = features['max'] - features['min']
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features[f'percentile_{p}'] = np.percentile(image, p)
        
        # Distribution shape
        flat_image = image.flatten()
        features['skewness'] = skew(flat_image)
        features['kurtosis'] = kurtosis(flat_image)
        
        # Energy and entropy
        features['energy'] = np.sum(image ** 2)
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        features['entropy'] = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        
        return features
    
    def extract_texture_features_glcm(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using Gray Level Co-occurrence Matrix (GLCM)
        
        Args:
            image: Grayscale image array (should be in range [0, 255])
            
        Returns:
            Dictionary of GLCM texture features
        """
        features = {}
        
        # Convert to uint8 if needed
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Calculate GLCM for different distances and angles
        for distance in self.glcm_distances:
            for angle_deg in self.glcm_angles:
                angle_rad = np.radians(angle_deg)
                
                try:
                    # Calculate GLCM
                    glcm = graycomatrix(
                        image_uint8, 
                        distances=[distance], 
                        angles=[angle_rad],
                        levels=256,
                        symmetric=True,
                        normed=True
                    )
                    
                    # Extract GLCM properties
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                    energy = graycoprops(glcm, 'energy')[0, 0]
                    correlation = graycoprops(glcm, 'correlation')[0, 0]
                    
                    # Store features with descriptive names
                    prefix = f'glcm_d{distance}_a{angle_deg}'
                    features[f'{prefix}_contrast'] = contrast
                    features[f'{prefix}_dissimilarity'] = dissimilarity
                    features[f'{prefix}_homogeneity'] = homogeneity
                    features[f'{prefix}_energy'] = energy
                    features[f'{prefix}_correlation'] = correlation
                    
                except Exception as e:
                    # Handle any GLCM calculation errors
                    prefix = f'glcm_d{distance}_a{angle_deg}'
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                        features[f'{prefix}_{prop}'] = 0.0
        
        # Calculate average GLCM features across all distances and angles
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in glcm_props:
            prop_values = [v for k, v in features.items() if prop in k]
            if prop_values:
                features[f'glcm_avg_{prop}'] = np.mean(prop_values)
                features[f'glcm_std_{prop}'] = np.std(prop_values)
        
        return features
    
    def extract_texture_features_lbp(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using Local Binary Patterns (LBP)
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary of LBP texture features
        """
        features = {}
        
        # Convert to uint8 if needed
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        try:
            # Calculate LBP
            lbp = local_binary_pattern(
                image_uint8, 
                self.lbp_n_points, 
                self.lbp_radius,
                method='uniform'
            )
            
            # Calculate LBP histogram
            n_bins = self.lbp_n_points + 2  # +2 for uniform patterns
            lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)  # Normalize
            
            # Extract LBP features
            features['lbp_uniformity'] = np.sum(lbp_hist ** 2)
            features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
            features['lbp_mean'] = np.mean(lbp)
            features['lbp_var'] = np.var(lbp)
            
            # Add histogram bins as features (first 10 most significant)
            for i in range(min(10, len(lbp_hist))):
                features[f'lbp_hist_bin_{i}'] = lbp_hist[i]
                
        except Exception as e:
            # Handle LBP calculation errors
            features['lbp_uniformity'] = 0.0
            features['lbp_entropy'] = 0.0
            features['lbp_mean'] = 0.0
            features['lbp_var'] = 0.0
            for i in range(10):
                features[f'lbp_hist_bin_{i}'] = 0.0
        
        return features
    
    def extract_gradient_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract gradient-based features
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary of gradient features
        """
        features = {}
        
        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Gradient features
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        features['gradient_max'] = np.max(gradient_magnitude)
        features['gradient_energy'] = np.sum(gradient_magnitude**2)
        
        # Gradient direction features
        features['gradient_dir_mean'] = np.mean(gradient_direction)
        features['gradient_dir_std'] = np.std(gradient_direction)
        
        # Edge density (high gradient regions)
        edge_threshold = np.percentile(gradient_magnitude, 90)
        edge_pixels = gradient_magnitude > edge_threshold
        features['edge_density'] = np.sum(edge_pixels) / gradient_magnitude.size
        
        return features
    
    def extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        # Convert to binary image for morphological operations
        binary_image = image > np.mean(image)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        
        # Opening and closing
        opened = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Morphological features
        features['morph_open_ratio'] = np.sum(opened) / binary_image.size
        features['morph_close_ratio'] = np.sum(closed) / binary_image.size
        features['morph_open_close_diff'] = np.sum(closed) - np.sum(opened)
        
        # Erosion and dilation
        eroded = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=1)
        dilated = cv2.dilate(binary_image.astype(np.uint8), kernel, iterations=1)
        
        features['morph_erosion_ratio'] = np.sum(eroded) / binary_image.size
        features['morph_dilation_ratio'] = np.sum(dilated) / binary_image.size
        
        return features
    
    def extract_region_features(self, 
                              image: np.ndarray, 
                              segmentation_result: Dict) -> Dict[str, float]:
        """
        Extract features from segmented brain regions
        
        Args:
            image: Original grayscale image
            segmentation_result: K-means segmentation results
            
        Returns:
            Dictionary of region-based features
        """
        features = {}
        
        region_masks = segmentation_result['region_masks']
        
        for region_name, mask in region_masks.items():
            if np.any(mask):
                region_pixels = image[mask]
                region_prefix = f'region_{region_name.lower().replace(" ", "_")}'
                
                # Basic statistics for each region
                features[f'{region_prefix}_mean'] = np.mean(region_pixels)
                features[f'{region_prefix}_std'] = np.std(region_pixels)
                features[f'{region_prefix}_area'] = np.sum(mask)
                features[f'{region_prefix}_area_ratio'] = np.sum(mask) / mask.size
                
                # Region shape features
                if np.sum(mask) > 10:  # Only if region has sufficient pixels
                    # Find contours
                    mask_uint8 = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Contour features
                        area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)
                        
                        features[f'{region_prefix}_contour_area'] = area
                        features[f'{region_prefix}_contour_perimeter'] = perimeter
                        features[f'{region_prefix}_compactness'] = (perimeter**2) / (4 * np.pi * area + 1e-8)
            else:
                # Empty region - set default values
                region_prefix = f'region_{region_name.lower().replace(" ", "_")}'
                for suffix in ['_mean', '_std', '_area', '_area_ratio', '_contour_area', '_contour_perimeter', '_compactness']:
                    features[f'{region_prefix}{suffix}'] = 0.0
        
        return features
    
    def extract_all_features(self, 
                           image: np.ndarray, 
                           segmentation_result: Dict = None) -> Dict[str, float]:
        """
        Extract all features from an image
        
        Args:
            image: Grayscale image array
            segmentation_result: Optional segmentation results for region features
            
        Returns:
            Dictionary containing all extracted features
        """
        all_features = {}
        
        # Statistical features
        stat_features = self.extract_statistical_features(image)
        all_features.update({f'stat_{k}': v for k, v in stat_features.items()})
        
        # Texture features - GLCM
        glcm_features = self.extract_texture_features_glcm(image)
        all_features.update(glcm_features)
        
        # Texture features - LBP
        lbp_features = self.extract_texture_features_lbp(image)
        all_features.update(lbp_features)
        
        # Gradient features
        gradient_features = self.extract_gradient_features(image)
        all_features.update({f'grad_{k}': v for k, v in gradient_features.items()})
        
        # Morphological features
        morph_features = self.extract_morphological_features(image)
        all_features.update(morph_features)
        
        # Region features (if segmentation provided)
        if segmentation_result is not None:
            region_features = self.extract_region_features(image, segmentation_result)
            all_features.update(region_features)
        
        return all_features