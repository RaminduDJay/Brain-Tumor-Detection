import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATASET_PATH = PROJECT_ROOT / "dataset"
    TRAINING_PATH = DATASET_PATH / "Training"
    TESTING_PATH = DATASET_PATH / "Testing"
    
    # Model paths
    MODELS_PATH = PROJECT_ROOT / "models"
    SAVED_MODELS_PATH = MODELS_PATH / "saved_models"
    CHECKPOINTS_PATH = MODELS_PATH / "checkpoints"
    
    # Results paths
    RESULTS_PATH = PROJECT_ROOT / "results"
    FIGURES_PATH = RESULTS_PATH / "figures"
    REPORTS_PATH = RESULTS_PATH / "reports"
    LOGS_PATH = RESULTS_PATH / "logs"
    
    # Class names
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Image parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Training parameters
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Flask app settings
    UPLOAD_FOLDER = PROJECT_ROOT / "app" / "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.MODELS_PATH,
            cls.SAVED_MODELS_PATH,
            cls.CHECKPOINTS_PATH,
            cls.RESULTS_PATH,
            cls.FIGURES_PATH,
            cls.REPORTS_PATH,
            cls.LOGS_PATH,
            cls.UPLOAD_FOLDER
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print("✅ All directories created successfully!")
