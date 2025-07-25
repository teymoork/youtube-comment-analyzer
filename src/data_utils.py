"""
Data utility functions for the new consolidated data architecture.
Handles loading, saving, and updating the canonical appdata files.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from pydantic import ValidationError

from src.logger_config import app_logger
from src.schemas import ChannelData, StoredVideoData, StoredComment

def load_app_data(app_data_path: Path) -> Optional[ChannelData]:
    """
    Loads the application's canonical data file for a channel.

    Args:
        app_data_path: The path to the appdata_*.json file.

    Returns:
        A ChannelData object if the file exists and is valid, otherwise None.
    """
    if not app_data_path.exists():
        app_logger.warning(f"App data file not found: {app_data_path}. An update from a source file is required.")
        return None
    
    app_logger.info(f"Loading canonical app data from: {app_data_path}")
    try:
        with open(app_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        channel_data = ChannelData.model_validate(data)
        app_logger.success(f"Successfully loaded and validated app data for '{channel_data.channel_metadata.title}'.")
        return channel_data
    except (json.JSONDecodeError, ValidationError) as e:
        app_logger.error(f"Failed to load or validate app data from {app_data_path.name}. The file may be corrupt. Error: {e}")
        return None

def save_app_data(app_data_path: Path, data: ChannelData):
    """
    Saves the ChannelData object to the application's canonical data file.

    Args:
        app_data_path: The path to save the appdata_*.json file.
        data: The ChannelData object to save.
    """
    app_data_path.parent.mkdir(parents=True, exist_ok=True)
    app_logger.info(f"Saving app data to: {app_data_path}")
    try:
        with open(app_data_path, 'w', encoding='utf-8') as f:
            # Use model_dump_json for Pydantic objects to ensure proper serialization
            f.write(data.model_dump_json(indent=4))
        app_logger.success(f"Successfully saved app data to {app_data_path.name}")
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while saving app data: {e}")

def update_data_from_source(
    existing_data: Optional[ChannelData], 
    source_data_dict: Dict[str, Any]
) -> ChannelData:
    """
    Merges data from a raw source file into the app's canonical data object.

    This function intelligently adds new videos and comments from the source
    while preserving existing data and analysis results in the app_data.

    Args:
        existing_data: The current ChannelData object, or None if it's the first run.
        source_data_dict: The raw data loaded from an external source JSON file.

    Returns:
        An updated ChannelData object with the new information merged in.
    """
    app_logger.info("Starting update process from source file...")
    
    # Use Pydantic to parse the raw source data first
    source_channel = ChannelData.model_validate(source_data_dict)

    if existing_data is None:
        app_logger.success("No existing app data found. Using source data as the new baseline.")
        return source_channel

    # Start with the existing data as the base
    updated_data = existing_data.model_copy(deep=True)

    # Update metadata from the newer source
    updated_data.channel_metadata = source_channel.channel_metadata

    new_videos_count = 0
    new_comments_count = 0

    # Iterate through videos in the source data
    for video_id, source_video in source_channel.videos.items():
        if video_id not in updated_data.videos:
            # Video is entirely new, add it completely
            updated_data.videos[video_id] = source_video
            new_videos_count += 1
            new_comments_count += len(source_video.comments)
        else:
            # Video exists, check for new comments
            app_video = updated_data.videos[video_id]
            
            # Also update video metadata in case it changed (e.g., title)
            app_video.video_metadata = source_video.video_metadata

            for comment_id, source_comment in source_video.comments.items():
                if comment_id not in app_video.comments:
                    # Comment is new, add it
                    app_video.comments[comment_id] = source_comment
                    new_comments_count += 1
    
    updated_data.last_video_list_check_timestamp = datetime.now(timezone.utc)
    app_logger.success(f"Update complete. Merged {new_videos_count} new videos and {new_comments_count} new comments.")
    
    return updated_data