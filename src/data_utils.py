# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/data_utils.py
import json
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import ValidationError

from src.logger_config import app_logger
from src.schemas import ChannelData

def load_channel_data(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Loads a single channel's data from a JSON file.

    Args:
        file_path: The path to the input JSON file.

    Returns:
        A dictionary containing the raw channel data, or None if loading fails.
    """
    if not file_path.exists():
        app_logger.error(f"File not found: {file_path}")
        return None

    app_logger.info(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        app_logger.success(f"Successfully loaded raw data from {file_path.name}")
        return data
    except json.JSONDecodeError:
        app_logger.error(f"Failed to decode JSON from {file_path.name}.")
        return None
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while loading {file_path.name}: {e}")
        return None

def load_or_initialize_results(analysis_file_path: Path) -> Dict[str, Any]:
    """
    Loads existing analysis results from a separate file or initializes a new structure.

    Args:
        analysis_file_path: The path to the analysis results JSON file.

    Returns:
        A dictionary of existing or newly initialized analysis results.
    """
    if analysis_file_path.exists():
        app_logger.info(f"Found existing analysis file. Loading: {analysis_file_path.name}")
        try:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            app_logger.warning(f"Could not read existing analysis file {analysis_file_path.name}. Starting fresh. Error: {e}")
            return {}
    else:
        app_logger.info(f"No analysis file found for {analysis_file_path.name}. Initializing new results.")
        return {}


def save_analysis_results(output_path: Path, analysis_results: Dict[str, Any]):
    """
    Saves the analysis results to a separate JSON file.

    Args:
        output_path: The path to save the analysis JSON file.
        analysis_results: The dictionary of analysis results, keyed by video_id.
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app_logger.info(f"Saving analysis results to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=4)
        app_logger.success(f"Successfully saved analysis results to {output_path.name}")
    except IOError as e:
        app_logger.error(f"Failed to write analysis results to {output_path.name}: {e}")
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while saving results: {e}")