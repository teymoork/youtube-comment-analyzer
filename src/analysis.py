import logging
from typing import Dict, List, Optional, Any, Tuple

import torch
from transformers import (
    pipeline,
    Pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

from src.config import model_paths
from src.logger_config import app_logger

def load_analysis_pipelines() -> Optional[Tuple[Pipeline, Pipeline, Pipeline, Pipeline]]:
    """
    Loads and initializes all the Hugging Face pipelines needed for analysis.
    This version explicitly loads models and tokenizers before creating pipelines
    to ensure local files are used correctly.

    Returns:
        A tuple containing the four initialized pipelines in order:
        (persian_sentiment, translation, english_emotion, irony)
        Returns None if any model fails to load.
    """
    try:
        app_logger.info("Initializing analysis pipelines...")
        device = 0 if torch.cuda.is_available() else -1
        app_logger.info(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

        # --- 1. Persian Sentiment Analysis ---
        app_logger.info(f"Loading Persian sentiment model from: {model_paths.persian_sentiment}")
        p_sent_path = str(model_paths.persian_sentiment)
        p_sent_tokenizer = AutoTokenizer.from_pretrained(p_sent_path, local_files_only=True)
        p_sent_model = AutoModelForSequenceClassification.from_pretrained(p_sent_path, local_files_only=True)
        persian_sentiment_pipeline = pipeline(
            "text-classification",
            model=p_sent_model,
            tokenizer=p_sent_tokenizer,
            return_all_scores=True,
            device=device,
        )

        # --- 2. Persian to English Translation ---
        app_logger.info(f"Loading translation model from: {model_paths.translation}")
        trans_path = str(model_paths.translation)
        trans_tokenizer = AutoTokenizer.from_pretrained(trans_path, local_files_only=True)
        trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_path, local_files_only=True)
        translation_pipeline = pipeline(
            "translation",
            model=trans_model,
            tokenizer=trans_tokenizer,
            src_lang="fa_IR",
            tgt_lang="en_XX",
            device=device,
        )

        # --- 3. English Multi-Emotion Analysis ---
        app_logger.info(f"Loading English emotion model from: {model_paths.english_emotion}")
        e_emo_path = str(model_paths.english_emotion)
        e_emo_tokenizer = AutoTokenizer.from_pretrained(e_emo_path, local_files_only=True)
        e_emo_model = AutoModelForSequenceClassification.from_pretrained(e_emo_path, local_files_only=True)
        english_emotion_pipeline = pipeline(
            "text-classification",
            model=e_emo_model,
            tokenizer=e_emo_tokenizer,
            return_all_scores=True,
            device=device,
        )

        # --- 4. English Irony Detection ---
        app_logger.info(f"Loading English irony model from: {model_paths.english_irony}")
        e_irony_path = str(model_paths.english_irony)
        e_irony_tokenizer = AutoTokenizer.from_pretrained(e_irony_path, local_files_only=True)
        e_irony_model = AutoModelForSequenceClassification.from_pretrained(e_irony_path, local_files_only=True)
        irony_pipeline = pipeline(
            "text-classification",
            model=e_irony_model,
            tokenizer=e_irony_tokenizer,
            device=device,
        )
        
        app_logger.success("All analysis models loaded successfully.")
        return (
            persian_sentiment_pipeline,
            translation_pipeline,
            english_emotion_pipeline,
            irony_pipeline,
        )

    except Exception as e:
        app_logger.critical(f"Failed to load one or more AI models. Error: {e}", exc_info=True)
        return None

def _process_scores(raw_output: List[Dict[str, Any]]) -> Dict[str, float]:
    """Helper to convert list of dicts to a single dict of label:score."""
    if not raw_output or not isinstance(raw_output, list) or not raw_output[0]:
        return {}
    # Some models wrap results in an extra list
    data_to_process = raw_output[0] if isinstance(raw_output[0], list) else raw_output
    return {item["label"]: round(item["score"], 4) for item in data_to_process}

def analyze_persian_emotion(text: str, pipeline: Pipeline) -> Optional[Dict[str, float]]:
    """Analyzes Persian text for sentiment, returning a dictionary of scores."""
    try:
        result = pipeline(text, top_k=None)
        return _process_scores(result)
    except Exception as e:
        app_logger.error(f"Persian emotion analysis failed for text: '{text[:50]}...'. Error: {e}")
        return None

def translate_text(text: str, pipeline: Pipeline) -> Optional[str]:
    """Translates Persian text to English."""
    try:
        result = pipeline(text)
        return result[0]["translation_text"]
    except Exception as e:
        app_logger.error(f"Translation failed for text: '{text[:50]}...'. Error: {e}")
        return None

def analyze_english_emotion(text: str, pipeline: Pipeline) -> Optional[Dict[str, float]]:
    """Analyzes English text for emotions, returning a dictionary of scores."""
    try:
        result = pipeline(text, top_k=None)
        return _process_scores(result)
    except Exception as e:
        app_logger.error(f"English emotion analysis failed for text: '{text[:50]}...'. Error: {e}")
        return None

def analyze_irony(text: str, pipeline: Pipeline) -> Optional[Dict[str, Any]]:
    """Analyzes English text for irony."""
    try:
        result = pipeline(text)
        # The irony model returns a list with a single dictionary
        return result[0] if result else None
    except Exception as e:
        app_logger.error(f"Irony analysis failed for text: '{text[:50]}...'. Error: {e}")
        return None