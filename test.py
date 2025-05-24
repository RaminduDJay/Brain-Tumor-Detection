# Run this first to create directories and validate setup
from config import Config
from utils.helpers import create_directory_structure, validate_dataset_structure

# Create directories
create_directory_structure()

# Validate dataset
validate_dataset_structure(Config.DATASET_PATH)