import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import List, Dict

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(log_file: str = "brain_tumor_detection.log"):
    """
    Setup logging configuration
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directory_structure():
    """
    Create the complete directory structure for the project
    """
    Config.create_directories()

def validate_dataset_structure(dataset_path: Path) -> bool:
    """
    Validate that the dataset has the expected structure
    """
    required_paths = [
        dataset_path / "Training" / "glioma",
        dataset_path / "Training" / "meningioma", 
        dataset_path / "Training" / "notumor",
        dataset_path / "Training" / "pituitary",
        dataset_path / "Testing" / "glioma",
        dataset_path / "Testing" / "meningioma",
        dataset_path / "Testing" / "notumor", 
        dataset_path / "Testing" / "pituitary"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    
    if missing_paths:
        print(f"❌ Missing directories: {missing_paths}")
        return False
    
    print("✅ Dataset structure validation passed!")
    return True

def get_system_info() -> Dict:
    """
    Get system information for reproducibility
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'tensorflow_version': tf.__version__,
        'gpu_available': tf.config.list_physical_devices('GPU')
    }