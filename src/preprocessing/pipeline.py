from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import joblib
import json
from .image_preprocessor import MRIImagePreprocessor
from .segmentation import BrainRegionSegmentation
from .feature_extractor import ComprehensiveFeatureExtractor
from .augmentation import MedicalImageAugmentation
from config import Config

class ComprehensivePreprocessingPipeline:
    """
    Complete preprocessing pipeline combining all preprocessing steps
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 enable_augmentation: bool = True,
                 augmentation_factor: int = 2):
        """
        Initialize the complete preprocessing pipeline
        
        Args:
            target_size: Target image dimensions
            enable_augmentation: Whether to apply data augmentation
            augmentation_factor: Number of augmented versions per image
        """
        self.target_size = target_size
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor
        
        # Initialize components
        self.preprocessor = MRIImagePreprocessor(target_size=target_size)
        self.segmentation = BrainRegionSegmentation()
        self.feature_extractor = ComprehensiveFeatureExtractor()
        
        if enable_augmentation:
            self.augmentation = MedicalImageAugmentation()
        
    def process_dataset(self, 
                       image_paths: List[str], 
                       labels: List[str],
                       save_processed: bool = True) -> Dict:
        """
        Process entire dataset through the complete pipeline
        
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            save_processed: Whether to save processed data
            
        Returns:
            Dictionary containing processed data and features
        """
        processed_images = []
        processed_labels = []
        segmentation_results = []
        feature_vectors = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} images")
            
            # Preprocess image
            result = self.preprocessor.preprocess_single_image(img_path)
            
            if result['success']:
                processed_img = result['processed_image']
                processed_images.append(processed_img)
                processed_labels.append(label)
                
                # Perform segmentation
                seg_result = self.segmentation.perform_segmentation(processed_img)
                segmentation_results.append(seg_result)
                
                # Extract features
                features = self.feature_extractor.extract_all_features(
                    processed_img, seg_result
                )
                feature_vectors.append(features)
        
        # Apply augmentation if enabled
        if self.enable_augmentation:
            print("Applying data augmentation...")
            aug_images, aug_labels = self.augmentation.create_augmented_dataset(
                processed_images, processed_labels, self.augmentation_factor
            )
            
            # Process augmented images for features
            aug_segmentation_results = []
            aug_feature_vectors = []
            
            for aug_img, aug_label in zip(aug_images, aug_labels):
                # Skip original images (already processed)
                if any(np.array_equal(aug_img, orig_img) for orig_img in processed_images):
                    continue
                
                # Segmentation for augmented image
                aug_seg_result = self.segmentation.perform_segmentation(aug_img)
                aug_segmentation_results.append(aug_seg_result)
                
                # Features for augmented image
                aug_features = self.feature_extractor.extract_all_features(
                    aug_img, aug_seg_result
                )
                aug_feature_vectors.append(aug_features)
            
            # Combine original and augmented data
            all_images = processed_images + [img for img in aug_images 
                                           if not any(np.array_equal(img, orig_img) 
                                                    for orig_img in processed_images)]
            all_labels = processed_labels + [label for img, label in zip(aug_images, aug_labels)
                                           if not any(np.array_equal(img, orig_img) 
                                                    for orig_img in processed_images)]
            all_segmentation = segmentation_results + aug_segmentation_results
            all_features = feature_vectors + aug_feature_vectors
        else:
            all_images = processed_images
            all_labels = processed_labels  
            all_segmentation = segmentation_results
            all_features = feature_vectors
        
        # Prepare output
        output = {
            'processed_images': np.array(all_images),
            'labels': all_labels,
            'segmentation_results': all_segmentation,
            'feature_vectors': all_features,
            'preprocessing_config': {
                'target_size': self.target_size,
                'augmentation_enabled': self.enable_augmentation,
                'augmentation_factor': self.augmentation_factor,
                'total_images': len(all_images)
            }
        }
        
        # Save processed data if requested
        if save_processed:
            self.save_processed_data(output)
        
        return output
    
    def save_processed_data(self, processed_data: Dict):
        """
        Save processed data to disk
        
        Args:
            processed_data: Dictionary containing processed data
        """
        # Create processed data directory
        processed_dir = Config.PROJECT_ROOT / "processed_data"
        processed_dir.mkdir(exist_ok=True)
        
        # Save processed images
        np.save(processed_dir / "processed_images.npy", processed_data['processed_images'])
        
        # Save labels
        with open(processed_dir / "labels.json", 'w') as f:
            json.dump(processed_data['labels'], f)
        
        # Save feature vectors
        joblib.dump(processed_data['feature_vectors'], processed_dir / "feature_vectors.pkl")
        
        # Save configuration
        with open(processed_dir / "preprocessing_config.json", 'w') as f:
            json.dump(processed_data['preprocessing_config'], f, indent=2)
        
        print(f"✅ Processed data saved to {processed_dir}")
    
    def load_processed_data(self) -> Dict:
        """
        Load previously processed data from disk
        
        Returns:
            Dictionary containing processed data
        """
        processed_dir = Config.PROJECT_ROOT / "processed_data"
        
        if not processed_dir.exists():
            raise FileNotFoundError("No processed data found. Run processing first.")
        
        # Load data
        processed_images = np.load(processed_dir / "processed_images.npy")
        
        with open(processed_dir / "labels.json", 'r') as f:
            labels = json.load(f)
        
        feature_vectors = joblib.load(processed_dir / "feature_vectors.pkl")
        
        with open(processed_dir / "preprocessing_config.json", 'r') as f:
            config = json.load(f)
        
        return {
            'processed_images': processed_images,
            'labels': labels,
            'feature_vectors': feature_vectors,
            'preprocessing_config': config
        }