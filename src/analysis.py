# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/analysis.py
from typing import Dict, Any, Optional, Tuple, List

import torch
from transformers import pipeline, Pipeline
from src.config import (
    PERSIAN_EMOTION_MODEL_ID,
    TRANSLATION_MODEL_ID,
    ENGLISH_EMOTION_MODEL_ID,
    IRONY_MODEL_ID,
)
from src.logger_config import app_logger

def load_analysis_pipelines() -> Optional[Tuple[Pipeline, Pipeline, Pipeline, Pipeline]]:
    """
    Loads and initializes all four Hugging Face pipelines.

    This function detects if a GPU is available and configures the pipelines
    to use it for better performance.

    Returns:
        A tuple containing the initialized pipelines in the order:
        (persian_emotion, translation, english_emotion, irony),
        or None if loading fails.
    """
    try:
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        app_logger.info(f"Loading analysis models. Using device: {device_name}")

        # 1. Persian Emotion Analysis Pipeline
        app_logger.info(f"Loading Persian emotion model: {PERSIAN_EMOTION_MODEL_ID}")
        persian_emotion_pipeline = pipeline(
            "text-classification",
            model=PERSIAN_EMOTION_MODEL_ID,
            device=device,
            return_all_scores=True
        )
        app_logger.success("Persian emotion model loaded successfully.")

        # 2. Translation Pipeline
        app_logger.info(f"Loading translation model: {TRANSLATION_MODEL_ID}")
        translation_pipeline = pipeline(
            "translation",
            model=TRANSLATION_MODEL_ID,
            device=device
        )
        app_logger.success("Translation model loaded successfully.")

        # 3. English Emotion Analysis Pipeline
        app_logger.info(f"Loading English emotion model: {ENGLISH_EMOTION_MODEL_ID}")
        english_emotion_pipeline = pipeline(
            "text-classification",
            model=ENGLISH_EMOTION_MODEL_ID,
            device=device,
            return_all_scores=True
        )
        app_logger.success("English emotion model loaded successfully.")

        # 4. Irony Detection Pipeline
        app_logger.info(f"Loading irony model: {IRONY_MODEL_ID}")
        irony_pipeline = pipeline(
            "text-classification",
            model=IRONY_MODEL_ID,
            device=device
        )
        app_logger.success("Irony model loaded successfully.")

        return persian_emotion_pipeline, translation_pipeline, english_emotion_pipeline, irony_pipeline

    except Exception as e:
        app_logger.error(f"Failed to load Hugging Face models. Error: {e}")
        app_logger.error("Please ensure you have a stable internet connection and correct model IDs.")
        return None

def analyze_persian_emotion(text: str, pipeline: Pipeline) -> Optional[Dict[str, float]]:
    """Analyzes a given Persian text for emotions."""
    if not text or not isinstance(text, str): return None
    try:
        results = pipeline(text)
        if results and isinstance(results, list) and len(results) > 0:
            return {item['label']: round(item['score'], 4) for item in results[0]}
        return None
    except Exception as e:
        app_logger.error(f"An error occurred during Persian emotion analysis: {e}")
        return None

def translate_text(text: str, pipeline: Pipeline) -> Optional[str]:
    """Translates Persian text to English, handling longer inputs."""
    if not text or not isinstance(text, str): return None
    try:
        # Set a higher max_length to handle longer comments. 512 is a safe default.
        results = pipeline(text, max_length=512)
        if results and isinstance(results, list) and 'translation_text' in results[0]:
            return results[0]['translation_text']
        return None
    except Exception as e:
        app_logger.error(f"An error occurred during translation: {e}")
        return None

def analyze_english_emotion(text: str, pipeline: Pipeline) -> Optional[Dict[str, float]]:
    """Analyzes a given English text for emotions."""
    if not text or not isinstance(text, str): return None
    try:
        results = pipeline(text)
        if results and isinstance(results, list) and len(results) > 0:
            return {item['label']: round(item['score'], 4) for item in results[0]}
        return None
    except Exception as e:
        app_logger.error(f"An error occurred during English emotion analysis: {e}")
        return None

def analyze_irony(text: str, pipeline: Pipeline) -> Optional[Dict[str, Any]]:
    """Analyzes a given English text for irony."""
    if not text or not isinstance(text, str): return None
    try:
        results = pipeline(text)
        if results and isinstance(results, list) and len(results) > 0:
            results[0]['score'] = round(results[0]['score'], 4)
            return results[0]
        return None
    except Exception as e:
        app_logger.error(f"An error occurred during irony analysis: {e}")
        return None