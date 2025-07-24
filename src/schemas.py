# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/emassch.py
"""
Defines the Pydantic models for structuring and validating YouTube data within the application.

This module serves as the single source of truth for the data shapes used when interacting
with the YouTube API, storing data in the central datastore (all_channels_data.json),
and potentially for other data processing tasks.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, HttpUrl, field_validator, RootModel

def _safe_int_cast(value: Any) -> int:
    """Safely casts a value to an integer, returning 0 on failure to match original model defaults."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

class BaseTimestampedModel(BaseModel):
    """
    A base Pydantic model that includes a `retrieved_at` timestamp.
    """
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("retrieved_at", mode="before")
    @classmethod
    def ensure_retrieved_at_utc(cls, value):
        """
        Pydantic validator to ensure `retrieved_at` is a UTC-aware datetime object.
        """
        if isinstance(value, str):
            try:
                dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid ISO format for retrieved_at")
        elif isinstance(value, datetime):
            dt_value = value
        else:
            raise TypeError("retrieved_at must be a datetime object or an ISO string")

        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

class StoredChannelMetadata(BaseTimestampedModel):
    """
    Represents the metadata for a YouTube channel that is stored by the application.
    """
    filename_stem: Optional[str] = Field(None, description="The unique, filesystem-safe identifier used for the channel's data file.", exclude=True)
    channel_id: str = Field(..., description="The unique ID of the YouTube channel.")
    title: Optional[str] = Field(None, description="The title of the channel.")
    sanitized_title: Optional[str] = Field(None, description="A sanitized version of the title, safe for filenames and display tables.")
    description: Optional[str] = Field(None, description="The description of the channel.")
    custom_url: Optional[str] = Field(None, description="The custom URL of the channel, if available.")
    published_at: Optional[datetime] = Field(None, description="The date and time that the channel was created.")
    country: Optional[str] = Field(None, description="The country associated with the channel, if available.")
    view_count: Optional[int] = Field(0, description="The total number of views for the channel.")
    subscriber_count: Optional[int] = Field(0, description="The total number of subscribers for the channel. Can be hidden by channel owner.")
    video_count: Optional[int] = Field(0, description="The total number of public videos uploaded by the channel.")
    
    uploads_playlist_id: Optional[str] = Field(None, description="The ID of the playlist containing all channel uploads.")

    last_metadata_update_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the last time the core metadata (title, desc, counts) was updated from API.")
    
    channel_sensuality: Optional[float] = Field(default=None, description="Aggregate sensuality score for the channel. (Total Emojis / Total Comment Chars).")
    avg_emojis_per_comment: Optional[float] = Field(default=None, description="Average number of emojis per comment across all comments on the channel.")
    avg_chars_per_comment: Optional[float] = Field(default=None, description="Average number of non-emoji characters per comment across all comments on the channel.")
    last_sensuality_calculation_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the last time sensuality metrics were calculated for this channel.")


    @field_validator("published_at", "last_metadata_update_timestamp", "last_sensuality_calculation_timestamp", mode="before")
    @classmethod
    def ensure_datetime_utc_optional(cls, value):
        """
        Pydantic validator for optional datetime fields to ensure they are UTC-aware.
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO format for datetime field: {value}")
        elif isinstance(value, datetime):
            dt_value = value
        else:
            raise TypeError("Datetime field must be a datetime object or an ISO string")

        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

class StoredComment(BaseModel):
    """
    Represents a single YouTube comment as stored by the application.
    """
    comment_id: str = Field(..., description="The unique ID of the comment.")
    text_original: Optional[str] = Field(None, description="The original text of the comment.")
    author_channel_id: Optional[str] = Field(None, description="The channel ID of the comment author.")
    author_display_name: Optional[str] = Field(None, description="The display name of the comment author.")
    published_at: datetime = Field(..., description="The date and time that the comment was published.")
    updated_at: datetime = Field(..., description="The date and time that the comment was last updated.")
    like_count: Optional[int] = Field(0, description="The number of likes received by the comment.")
    parent_id: Optional[str] = Field(None, description="The ID of the parent comment if this is a reply. Null for top-level comments.")
    total_reply_count: Optional[int] = Field(0, description="The total number of replies to this comment (if it's a top-level comment).")

    @field_validator("published_at", "updated_at", mode="before")
    @classmethod
    def ensure_comment_datetime_utc(cls, value):
        """
        Pydantic validator for comment datetime fields to ensure they are UTC-aware.
        """
        if isinstance(value, str):
            try:
                dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO format for comment datetime field: {value}")
        elif isinstance(value, datetime):
            dt_value = value
        else:
            raise TypeError("Comment datetime field must be a datetime object or an ISO string")

        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

    @field_validator("author_channel_id", mode="before")
    @classmethod
    def extract_author_channel_id_value(cls, v):
        """
        Normalizes the `author_channel_id` field from a potential dict to a string.
        """
        if isinstance(v, dict) and 'value' in v:
            return v['value']
        return v

class StoredVideoMetadata(BaseTimestampedModel):
    """
    Represents the metadata for a YouTube video that is stored by the application.
    """
    video_id: str = Field(..., description="The unique ID of the YouTube video.")
    title: Optional[str] = Field(None, description="The title of the video.")
    published_at: Optional[datetime] = Field(None, description="The date and time that the video was published.")
    view_count: Optional[int] = Field(0, description="The total number of views for the video.")
    like_count: Optional[int] = Field(0, description="The total number of likes for the video.")
    comment_count: Optional[int] = Field(0, description="The total number of comments for the video (as reported by video stats, may include replies).")
    duration_iso: Optional[str] = Field(None, description="The duration of the video in ISO 8601 format.")
    channel_id: str = Field(..., description="The ID of the channel that uploaded the video.")
    channel_title: Optional[str] = Field(None, description="The title of the channel that uploaded the video (denormalized).")
    tags: Optional[List[str]] = Field(default_factory=list, description="A list of tags associated with the video.")
    category_id: Optional[str] = Field(None, description="The YouTube category ID of the video.")
    url: Optional[HttpUrl] = Field(None, description="The URL to the video on YouTube.")
    last_metadata_update_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the last time core metadata (title, counts, etc.) was updated from API.")

    total_comment_chars: Optional[int] = Field(default=None, description="Sum of lengths of all comments, excluding emojis.")
    total_emojis: Optional[int] = Field(default=None, description="Total count of all emojis in comments.")
    avg_comment_chars: Optional[float] = Field(default=None, description="Average number of non-emoji characters per comment.")
    avg_emojis: Optional[float] = Field(default=None, description="Average number of emojis per comment.")
    video_sensuality: Optional[float] = Field(default=None, description="Sensuality score for the video. (Total Emojis / Total Comment Chars).")


    @field_validator("published_at", "last_metadata_update_timestamp", mode="before")
    @classmethod
    def ensure_video_datetime_utc_optional(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO format for video datetime field: {value}")
        elif isinstance(value, datetime):
            dt_value = value
        else:
            raise TypeError("Video datetime field must be a datetime object or an ISO string")

        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

    @field_validator("url", mode="before")
    @classmethod
    def ensure_url_str(cls, value):
        if value is None:
            return None
        return str(value)

class StoredVideoData(BaseModel):
    """
    Represents the comprehensive data for a single YouTube video stored by the application.
    """
    video_metadata: StoredVideoMetadata = Field(..., description="Core metadata for the video.")
    comments: Dict[str, StoredComment] = Field(default_factory=dict, description="A dictionary of comments for the video, keyed by comment_id.")
    last_comments_check_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the last time comments were checked/fetched for this video.")

    @field_validator("last_comments_check_timestamp", mode="before")
    @classmethod
    def ensure_comments_check_datetime_utc_optional(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO format for last_comments_check_timestamp: {value}")
        elif isinstance(value, datetime):
            dt_value = value
        else:
            raise TypeError("last_comments_check_timestamp must be a datetime object or an ISO string")

        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

class ChannelData(BaseModel):
    """
    Represents all stored data associated with a single YouTube channel.
    """
    channel_metadata: StoredChannelMetadata = Field(..., description="Core metadata for the channel.")
    videos: Dict[str, StoredVideoData] = Field(default_factory=dict, description="A dictionary of videos for this channel, keyed by video_id.")
    last_video_list_check_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the last time the video list was fetched for this channel.")

    @field_validator("last_video_list_check_timestamp", mode="before")
    @classmethod
    def ensure_video_list_check_datetime_utc_optional(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                dt_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO format for last_video_list_check_timestamp: {value}")
        elif isinstance(value, datetime):
            dt_value = value
        else:
            raise TypeError("last_video_list_check_timestamp must be a datetime object or an ISO string")

        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

class AllChannelsDataStore(RootModel[Dict[str, ChannelData]]):
    """
    The main datastore structure.
    """
    root: Dict[str, ChannelData] = Field(default_factory=dict, description="The root dictionary mapping channel IDs to ChannelData.")

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item: str) -> ChannelData:
        return self.root[item]

    def __setitem__(self, key: str, value: ChannelData):
        if not isinstance(value, ChannelData):
            if isinstance(value, dict):
                try:
                    value = ChannelData(**value)
                except Exception as e:
                    raise TypeError(f"Value for key '{key}' must be a ChannelData instance or a dict parsable into ChannelData. Error: {e}") from e
            else:
                raise TypeError(f"Value for key '{key}' must be a ChannelData instance or a dict. Got {type(value)}")
        self.root[key] = value

    def __len__(self) -> int:
        return len(self.root)

    def items(self):
        return self.root.items()

    def values(self):
        return self.root.values()

    def keys(self):
        return self.root.keys()

    def get(self, key: str, default: Optional[Any] = None) -> Optional[ChannelData]:
        return self.root.get(key, default)