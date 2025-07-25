"""
Configuration settings for the YouTube Comment Analyzer application.

This file defines paths to local models, data storage directories, and logging files.
It uses Pydantic for structured configuration management.
"""
import os
from pathlib import Path
from pydantic import BaseModel

# --- Core Paths ---

# Define the absolute path to the project's root directory.
# This is used to build other absolute paths, ensuring they work from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base path for all locally stored Hugging Face models.
# Path.home() correctly resolves the '~' user home directory character.
HUGGINGFACE_MODELS_DIR = Path.home() / "huggingface_models"

# Directory for raw input JSON files.
INPUT_DATA_DIR = PROJECT_ROOT / "input_data"

# Directory for processed analysis files.
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

# Path to the application log file.
LOG_FILE = PROJECT_ROOT / "app_log.txt"

# --- Application Settings ---

# Default number of comments to analyze per video if not specified otherwise.
DEFAULT_COMMENTS_PER_VIDEO = 2000

# Number of comments to process before saving a checkpoint to disk.
CHECKPOINT_INTERVAL = 10


# --- Pydantic Model for Model Paths ---

class ModelPaths(BaseModel):
    """A Pydantic model to hold the paths for all required ML models."""
    persian_sentiment: Path
    translation: Path
    english_emotion: Path
    english_irony: Path

# --- Application Configuration Instance ---

# Create a single, validated instance of the model paths for the application to use.
# This ensures all paths are defined and provides easy autocompletion in IDEs.
# NOTE: These paths have been corrected to match the user's local filesystem.
model_paths = ModelPaths(
    persian_sentiment=HUGGINGFACE_MODELS_DIR / "HooshvareLab-bert-fa-base-uncased-clf-persiannews",
    translation=HUGGINGFACE_MODELS_DIR / "persiannlp-mt5-base-translation",
    english_emotion=HUGGINGFACE_MODELS_DIR / "j-hartmann-emotion-english-distilroberta-base",
    english_irony=HUGGINGFACE_MODELS_DIR / "twitter-roberta-base-irony"
)

# --- Utility Functions ---

def ensure_data_dir_exists():
    """Creates the processed data directory if it doesn't already exist."""
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)