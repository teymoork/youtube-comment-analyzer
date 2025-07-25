# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/config.py
from pathlib import Path

# --- Home Directory ---
# Get the user's home directory in an OS-agnostic way
HOME_DIR = Path.home()

# --- Project Root ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# --- Data Directories ---
# Defines the paths to the input and processed data folders.
# Path to the external data collection project's output directory.
INPUT_DATA_DIR = Path("/home/tkh/repos/ml/graph-ml/youtube/data_collection/processed_data/channels/")
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

# --- Hugging Face Model Identifiers ---
# USE LOCAL PATHS TO BYPASS NETWORK ISSUES
TRANSLATION_MODEL_ID = HOME_DIR / "huggingface_models/persiannlp-mt5-base-translation"
PERSIAN_EMOTION_MODEL_ID = HOME_DIR / "huggingface_models/HooshvareLab-bert-fa-base-uncased-clf-persiannews"
ENGLISH_EMOTION_MODEL_ID = HOME_DIR / "huggingface_models/j-hartmann-emotion-english-distilroberta-base"
IRONY_MODEL_ID = HOME_DIR / "huggingface_models/twitter-roberta-base-irony"


# --- Analysis Results ---
# This is the key that will be used to store the analysis results
# within the processed JSON file for each channel.
ANALYSIS_RESULTS_KEY = "analysis_results"

# --- Interactive Mode Defaults ---
# Default number of comments to process per video in interactive mode.
DEFAULT_COMMENTS_PER_VIDEO = 10000