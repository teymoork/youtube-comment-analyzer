# /home/tkh/repos/hugging_face/youtube_comment_analyzer/src/aggregation.py
"""
This module contains functions for calculating aggregate analysis statistics
for videos and channels.
"""
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from src.schemas import StoredVideoData, AggregateAnalysis, ChannelData


def calculate_video_aggregates(video_data: StoredVideoData) -> Optional[AggregateAnalysis]:
    """
    Calculates aggregate analysis scores based on all analyzed comments within a single video.

    Args:
        video_data: The StoredVideoData object containing comments for one video.

    Returns:
        An AggregateAnalysis object with the computed averages, or None if no
        comments have been analyzed for this video.
    """
    analyzed_comments = [
        c for c in video_data.comments.values() if c.analysis is not None
    ]

    if not analyzed_comments:
        return None

    total_comments = len(analyzed_comments)
    
    # Initialize accumulators
    persian_sentiment_sums = defaultdict(float)
    english_emotion_sums = defaultdict(float)
    irony_counts = defaultdict(int)
    
    for comment in analyzed_comments:
        # Aggregate Persian Sentiment
        if comment.analysis.persian_sentiment:
            for label, score in comment.analysis.persian_sentiment.items():
                persian_sentiment_sums[label] += score

        # Aggregate English Emotions
        if comment.analysis.english_emotions:
            for label, score in comment.analysis.english_emotions.items():
                english_emotion_sums[label] += score

        # Count Irony labels
        if comment.analysis.english_irony and comment.analysis.english_irony.label:
            irony_counts[comment.analysis.english_irony.label] += 1

    # Calculate averages for sentiment and emotion scores
    avg_persian_sentiment = {
        label: round(total / total_comments, 4)
        for label, total in persian_sentiment_sums.items()
    }
    
    avg_english_emotions = {
        label: round(total / total_comments, 4)
        for label, total in english_emotion_sums.items()
    }

    # Calculate distribution (percentage) for irony labels
    total_irony_comments = sum(irony_counts.values())
    irony_distribution = {}
    if total_irony_comments > 0:
        irony_distribution = {
            label: round(count / total_irony_comments, 4)
            for label, count in irony_counts.items()
        }

    return AggregateAnalysis(
        total_analyzed_comments=total_comments,
        avg_persian_sentiment=avg_persian_sentiment or None,
        avg_english_emotions=avg_english_emotions or None,
        irony_distribution=irony_distribution or None,
        last_calculated_at=datetime.now(timezone.utc)
    )


def calculate_channel_aggregates(channel_data: ChannelData) -> Optional[AggregateAnalysis]:
    """
    Calculates aggregate analysis scores for an entire channel by aggregating
    the pre-calculated scores from each video. This is more efficient than
    re-processing every comment.

    Args:
        channel_data: The main ChannelData object.

    Returns:
        An AggregateAnalysis object with the computed averages for the channel,
        or None if no videos have aggregate analysis.
    """
    videos_with_aggregates = [
        v.video_metadata for v in channel_data.videos.values() 
        if v.video_metadata.aggregate_analysis is not None
    ]

    if not videos_with_aggregates:
        return None

    total_analyzed_comments_in_channel = 0
    
    # Accumulators for weighted averages
    persian_sentiment_weighted_sums = defaultdict(float)
    english_emotion_weighted_sums = defaultdict(float)
    irony_weighted_sums = defaultdict(float)

    for video_meta in videos_with_aggregates:
        agg = video_meta.aggregate_analysis
        num_comments_in_video = agg.total_analyzed_comments
        total_analyzed_comments_in_channel += num_comments_in_video

        # Add weighted scores for Persian sentiment
        if agg.avg_persian_sentiment:
            for label, avg_score in agg.avg_persian_sentiment.items():
                persian_sentiment_weighted_sums[label] += avg_score * num_comments_in_video

        # Add weighted scores for English emotions
        if agg.avg_english_emotions:
            for label, avg_score in agg.avg_english_emotions.items():
                english_emotion_weighted_sums[label] += avg_score * num_comments_in_video
        
        # Add weighted counts for irony distribution
        if agg.irony_distribution:
            for label, dist in agg.irony_distribution.items():
                irony_weighted_sums[label] += dist * num_comments_in_video

    if total_analyzed_comments_in_channel == 0:
        return None

    # Calculate final channel averages
    final_avg_persian_sentiment = {
        label: round(total_score / total_analyzed_comments_in_channel, 4)
        for label, total_score in persian_sentiment_weighted_sums.items()
    }

    final_avg_english_emotions = {
        label: round(total_score / total_analyzed_comments_in_channel, 4)
        for label, total_score in english_emotion_weighted_sums.items()
    }

    # Calculate final channel irony distribution
    final_irony_distribution = {
        label: round(total_count / total_analyzed_comments_in_channel, 4)
        for label, total_count in irony_weighted_sums.items()
    }

    return AggregateAnalysis(
        total_analyzed_comments=total_analyzed_comments_in_channel,
        avg_persian_sentiment=final_avg_persian_sentiment or None,
        avg_english_emotions=final_avg_english_emotions or None,
        irony_distribution=final_irony_distribution or None,
        last_calculated_at=datetime.now(timezone.utc)
    )