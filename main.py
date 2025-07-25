# /home/tkh/repos/hugging_face/youtube_comment_analyzer/main.py
import typer
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from pydantic import ValidationError
import json
from datetime import datetime, timezone
import time
import math

# Import our own modules
from src.logger_config import app_logger
from src.config import (
    INPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_COMMENTS_PER_VIDEO,
    CHECKPOINT_INTERVAL,
)
from src.schemas import ChannelData, AnalysisResult, StoredComment, StoredVideoData, IronyResult
from src.data_utils import (
    load_app_data,
    save_app_data,
    update_data_from_source,
)
from src.analysis import (
    load_analysis_pipelines,
    analyze_persian_emotion_batch,
    translate_text_batch,
    analyze_english_emotion_batch,
    analyze_irony_batch,
)
from src.aggregation import calculate_video_aggregates, calculate_channel_aggregates
from transformers import Pipeline

# Initialize Typer app and Rich console
app = typer.Typer(help="A CLI tool to analyze YouTube comments with a multi-stage pipeline.")
console = Console()

def select_source_file() -> Optional[Path]:
    """Displays a menu of external source data files for the user to choose from."""
    console.print("\n[bold cyan]Select a channel source file to work with:[/bold cyan]")
    
    try:
        source_files = sorted([f for f in INPUT_DATA_DIR.glob('*.json') if f.is_file()])
    except FileNotFoundError:
        console.print(f"[bold red]Error: Source directory not found at '{INPUT_DATA_DIR}'[/bold red]")
        return None

    if not source_files:
        console.print(f"[yellow]No channel JSON files found in '{INPUT_DATA_DIR}'.[/yellow]")
        return None

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan")
    table.add_column("Source Filename")
    for i, file_path in enumerate(source_files):
        table.add_row(str(i + 1), file_path.name)
    console.print(table)
    console.print("  [green]q[/green]: Quit")

    choice = Prompt.ask("Enter your choice", default="q")

    if choice.lower() == 'q':
        return None
    
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(source_files):
            return source_files[choice_index]
        else:
            console.print("[red]Invalid choice.[/red]")
            return select_source_file()
    except ValueError:
        console.print("[red]Invalid input.[/red]")
        return select_source_file()

def run_batch_analysis_on_comments(
    comments_to_analyze: List[StoredComment],
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
) -> int:
    """
    Performs a multi-stage batch analysis on a list of comments.
    
    Returns the number of comments successfully processed.
    """
    if not comments_to_analyze:
        return 0

    app_logger.info(f"Starting batch analysis for {len(comments_to_analyze)} comments.")

    MAX_COMMENT_CHARS = 512 # Use 512 as a safe limit for all models
    
    # 1. Prepare batch data for Persian analysis
    original_texts = []
    persian_texts_for_analysis = []
    for c in comments_to_analyze:
        text = c.text_original or ""
        original_texts.append(text)
        persian_texts_for_analysis.append(text[:MAX_COMMENT_CHARS])

    # 2. Run batch analysis pipelines
    persian_emotion_pipeline, translation_pipeline, english_emotion_pipeline, irony_pipeline = pipelines
    
    persian_results = analyze_persian_emotion_batch(persian_texts_for_analysis, persian_emotion_pipeline)
    translated_texts = translate_text_batch(persian_texts_for_analysis, translation_pipeline)
    
    # 3. Prepare texts for English analysis, truncating them as well
    english_texts_for_analysis = []
    english_analysis_indices = []
    for i, text in enumerate(translated_texts):
        if text:
            english_texts_for_analysis.append(text[:MAX_COMMENT_CHARS])
            english_analysis_indices.append(i)

    english_emotion_results_sparse = []
    irony_results_sparse = []
    if english_texts_for_analysis:
        english_emotion_results_sparse = analyze_english_emotion_batch(english_texts_for_analysis, english_emotion_pipeline)
        irony_results_sparse = analyze_irony_batch(english_texts_for_analysis, irony_pipeline)

    # Re-map sparse results back to the original comment list structure
    english_emotion_results = [None] * len(comments_to_analyze)
    irony_results = [None] * len(comments_to_analyze)
    for i, sparse_idx in enumerate(english_analysis_indices):
        english_emotion_results[sparse_idx] = english_emotion_results_sparse[i]
        irony_results[sparse_idx] = irony_results_sparse[i]

    # 4. Assign results back to comment objects
    processed_count = 0
    for i, comment in enumerate(comments_to_analyze):
        irony_obj = IronyResult.model_validate(irony_results[i]) if irony_results[i] else None

        analysis_result = AnalysisResult(
            persian_sentiment=persian_results[i],
            english_translation=translated_texts[i],
            english_emotions=english_emotion_results[i],
            english_irony=irony_obj,
            analyzed_at=datetime.now(timezone.utc)
        )
        comment.analysis = analysis_result
        processed_count += 1

        if verbose:
            console.print(f"\n[bold]Result for comment {i + 1}/{len(comments_to_analyze)} (ID: {comment.comment_id})[/]")
            console.print(f"[bold]Original Text:[/] '{original_texts[i][:100]}...'")
            console.print(f"[bold magenta]Persian Sentiment:[/] {analysis_result.persian_sentiment}")
            console.print("---")
            console.print(f"[bold]Translated Text:[/] '{analysis_result.english_translation[:100] if analysis_result.english_translation else 'N/A'}'")
            console.print(f"[bold cyan]English Emotions:[/] {analysis_result.english_emotions}")
            console.print(f"[bold yellow]Irony Detection:[/] {analysis_result.english_irony}")
            
    app_logger.success(f"Batch analysis complete. Processed {processed_count} comments.")
    return processed_count

def update_and_log_aggregates(channel_data: ChannelData):
    """Calculates and updates aggregates for all videos and the overall channel."""
    app_logger.info("Updating all aggregate statistics...")
    
    for video_data in channel_data.videos.values():
        video_agg = calculate_video_aggregates(video_data)
        if video_agg:
            video_data.video_metadata.aggregate_analysis = video_agg

    channel_agg = calculate_channel_aggregates(channel_data)
    if channel_agg:
        channel_data.channel_metadata.aggregate_analysis = channel_agg
    
    app_logger.success("Aggregate statistics updated.")

def process_comment_list_with_checkpoints(
    all_comments_to_process: List[StoredComment],
    channel_data: ChannelData,
    app_data_path: Path,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
) -> int:
    """
    Processes a list of comments in smaller chunks, saving after each chunk.
    Returns the total number of comments processed.
    """
    total_comments = len(all_comments_to_process)
    if total_comments == 0:
        return 0

    num_chunks = math.ceil(total_comments / CHECKPOINT_INTERVAL)
    console.print(f"[bold]Found {total_comments} new comments. Processing in {num_chunks} batches of up to {CHECKPOINT_INTERVAL}.[/bold]")

    processed_count_total = 0
    for i in range(num_chunks):
        start_index = i * CHECKPOINT_INTERVAL
        end_index = start_index + CHECKPOINT_INTERVAL
        comment_chunk = all_comments_to_process[start_index:end_index]
        
        console.print(
            f"\n--- Processing batch {i + 1}/{num_chunks} (comments {start_index + 1}-{min(end_index, total_comments)} of {total_comments}) ---"
        )
        
        start_time = time.perf_counter()
        processed_in_chunk = run_batch_analysis_on_comments(comment_chunk, pipelines, verbose)
        end_time = time.perf_counter()
        
        if processed_in_chunk > 0:
            console.print(f"  [green]Batch processed in {end_time - start_time:.2f} seconds.[/green]")
            processed_count_total += processed_in_chunk
            
            console.print("  [bold]Creating checkpoint: Updating aggregates and saving to disk...[/bold]")
            update_and_log_aggregates(channel_data)
            save_app_data(app_data_path, channel_data)
            console.print("  [green]Checkpoint saved successfully.[/green]")

    console.print(f"\n[green]Finished analysis for this set. Processed {processed_count_total} new comments.[/green]")
    return processed_count_total

def handle_batch_analysis(
    channel_data: ChannelData,
    app_data_path: Path,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
):
    """Gathers comments for batch mode and processes them video by video."""
    num_comments_per_video = IntPrompt.ask(
        f"How many new comments to analyze per video?",
        default=DEFAULT_COMMENTS_PER_VIDEO
    )
    num_videos_to_process = IntPrompt.ask(
        "How many videos to process?",
        default=len(channel_data.videos)
    )
    
    videos_to_process = list(channel_data.videos.values())[:num_videos_to_process]
    total_processed_in_session = 0

    for video_index, video_data in enumerate(videos_to_process):
        console.print(
            f"\n[bold cyan]>>> Processing Video {video_index + 1}/{len(videos_to_process)}: '{video_data.video_metadata.title}' <<<[/bold cyan]"
        )
        
        comments_for_this_video = [c for c in video_data.comments.values() if c.analysis is None]
        sorted_comments = sorted(comments_for_this_video, key=lambda c: c.published_at, reverse=True)
        comments_for_this_run = sorted_comments[:num_comments_per_video]
        
        if not comments_for_this_run:
            console.print("[yellow]No new comments to analyze in this video.[/yellow]")
            continue
        
        processed_count = process_comment_list_with_checkpoints(
            comments_for_this_run, channel_data, app_data_path, pipelines, verbose
        )
        total_processed_in_session += processed_count

    console.print(f"\n[bold blue]>>> Batch Mode Finished. Analyzed a total of {total_processed_in_session} new comments across {len(videos_to_process)} videos checked. <<<[/bold blue]")

def handle_video_analysis(
    channel_data: ChannelData,
    app_data_path: Path,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
):
    """Gathers comments for a single video and passes them to the checkpoint processor."""
    table = Table(title="Videos in Channel")
    table.add_column("Index", style="cyan")
    table.add_column("Video Title", style="magenta")
    table.add_column("Video ID", style="green")

    video_list = list(channel_data.videos.values())
    for i, video_data in enumerate(video_list):
        table.add_row(str(i + 1), video_data.video_metadata.title, video_data.video_metadata.video_id)
    
    console.print(table)
    choice = IntPrompt.ask("Select a video to analyze", default=1)
    
    try:
        selected_video_data = video_list[choice - 1]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/red]")
        return

    all_comments_to_process = [c for c in selected_video_data.comments.values() if c.analysis is None]
    
    processed_count = process_comment_list_with_checkpoints(
        all_comments_to_process, channel_data, app_data_path, pipelines, verbose
    )
    
    console.print(f"\n[bold blue]>>> Video Analysis Finished. Analyzed a total of {processed_count} new comments for this video. <<<[/bold blue]")

def display_channel_stats(channel_data: Optional[ChannelData]):
    """Displays statistics about the loaded channel data."""
    if not channel_data:
        console.print("\n[yellow]No application data loaded. Please update from a source file first.[/yellow]")
        return

    total_videos = len(channel_data.videos)
    total_comments = 0
    analyzed_comments = 0
    for video in channel_data.videos.values():
        video_comments = len(video.comments)
        total_comments += video_comments
        for comment in video.comments.values():
            if comment.analysis:
                analyzed_comments += 1
    
    table = Table(title=f"Statistics for '{channel_data.channel_metadata.title}'", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Videos", str(total_videos))
    table.add_row("Total Comments", str(total_comments))
    table.add_row("Analyzed Comments", str(analyzed_comments))
    table.add_row("Unanalyzed Comments", str(total_comments - analyzed_comments))
    
    update_time = channel_data.last_video_list_check_timestamp
    table.add_row("Last Source Update", update_time.strftime("%Y-%m-%d %H:%M:%S %Z") if update_time else "Never")
    
    console.print(table)

    if channel_data.channel_metadata.aggregate_analysis:
        console.print("\n[bold cyan]Channel Aggregate Analysis[/bold cyan]")
        agg_table = Table(show_header=True, header_style="bold magenta", title="Overall Channel Averages")
        agg_table.add_column("Metric", style="cyan")
        agg_table.add_column("Label", style="green")
        agg_table.add_column("Average Score", style="yellow", justify="right")

        agg_data = channel_data.channel_metadata.aggregate_analysis
        
        agg_table.add_row("Total Analyzed", "-", str(agg_data.total_analyzed_comments), end_section=True)

        if agg_data.avg_persian_sentiment:
            for label, score in sorted(agg_data.avg_persian_sentiment.items()):
                agg_table.add_row("Persian Sentiment", label, f"{score:.4f}")
            agg_table.add_row("", "", "", end_section=True)

        if agg_data.avg_english_emotions:
            for label, score in sorted(agg_data.avg_english_emotions.items()):
                agg_table.add_row("English Emotion", label, f"{score:.4f}")
            agg_table.add_row("", "", "", end_section=True)

        if agg_data.irony_distribution:
            for label, dist in sorted(agg_data.irony_distribution.items()):
                agg_table.add_row("Irony Distribution", label, f"{dist:.2%}")

        console.print(agg_table)
    else:
        console.print("\n[yellow]No aggregate analysis data available for this channel. Run an analysis to generate it.[/yellow]")

def display_data_health_check(source_file: Path, channel_data: Optional[ChannelData]):
    """Compares the source file with the app data to show data status."""
    console.print("\n[bold cyan]Data Health Check[/bold cyan]")
    
    source_comments = 0
    try:
        with source_file.open('r', encoding='utf-8') as f:
            source_dict = json.load(f)
        for video in source_dict.get('videos', {}).values():
            source_comments += len(video.get('comments', {}))
    except (FileNotFoundError, json.JSONDecodeError):
        console.print(f"[red]Could not read or parse source file: {source_file.name}[/red]")
        return

    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Comments in Source File", f"[bold green]{source_comments}[/bold green]")

    if channel_data:
        app_total_comments = 0
        app_unanalyzed_comments = 0
        for video in channel_data.videos.values():
            app_total_comments += len(video.comments)
            for comment in video.comments.values():
                if comment.analysis is None:
                    app_unanalyzed_comments += 1
        
        table.add_row("Comments in App Data", str(app_total_comments))
        table.add_row("Comments Needing Analysis", f"[bold yellow]{app_unanalyzed_comments}[/bold yellow]")
    else:
        table.add_row("Comments in App Data", "[red]0 (No app data loaded)[/red]")
        table.add_row("Comments Needing Analysis", "[red]N/A[/red]")

    console.print(table)
    
    if channel_data and source_comments > app_total_comments:
        console.print("\n[bold yellow]Warning:[/bold yellow] The source file has more comments than the app data.")
        console.print("Run '[bold green]1: Update from Source File[/bold green]' to load the new comments before analyzing.")
    elif not channel_data:
         console.print("\n[bold red]Error:[/bold red] No application data is loaded.")
         console.print("Run '[bold green]1: Update from Source File[/bold green]' to begin.")
    else:
        console.print("\n[green]Data is synchronized. You can proceed with analysis.[/green]")


def process_channel(source_file: Path, pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline], verbose: bool):
    """Main logic loop for managing and analyzing a single channel."""
    channel_stem = source_file.stem
    app_data_path = PROCESSED_DATA_DIR / f"appdata_{channel_stem}.json"
    
    channel_data = load_app_data(app_data_path)

    while True:
        console.print(f"\n[bold cyan]Managing Channel: {channel_stem}[/bold cyan]")
        console.print("  [green]1[/green]: Update from Source File")
        console.print("  [green]2[/green]: Analyze Comments (Batch Mode)")
        console.print("  [green]3[/green]: Analyze Comments (Specific Video)")
        console.print("  [green]4[/green]: View Channel Statistics")
        console.print("  [green]5[/green]: Data Health Check")
        console.print("  [green]b[/green]: Back to Main Menu")

        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "5", "b"], default="b")

        if choice == '1':
            console.print(f"\n[bold]Updating from source file: {source_file.name}...[/bold]")
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_dict = json.load(f)
                # Reload data before updating to ensure it's fresh
                channel_data = load_app_data(app_data_path)
                channel_data = update_data_from_source(channel_data, source_dict)
                save_app_data(app_data_path, channel_data)
                console.print("[green]Update and save complete.[/green]")
            except (json.JSONDecodeError, ValidationError, FileNotFoundError) as e:
                console.print(f"[bold red]Error during update: {e}[/bold red]")
        
        elif choice in ['2', '3']:
            if not channel_data:
                console.print("\n[bold red]Cannot analyze. No data loaded. Please run 'Update from Source' first.[/bold red]")
                continue
            
            if choice == '2':
                handle_batch_analysis(channel_data, app_data_path, pipelines, verbose)
            elif choice == '3':
                handle_video_analysis(channel_data, app_data_path, pipelines, verbose)

        elif choice == '4':
            display_channel_stats(channel_data)

        elif choice == '5':
            display_data_health_check(source_file, channel_data)

        elif choice.lower() == 'b':
            console.print("[yellow]Returning to main menu.[/yellow]")
            break

@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print analysis results to the screen.")
):
    """Starts the interactive YouTube Comment Analyzer tool."""
    console.print("[bold blue]--- YouTube Comment Analyzer ---[/bold blue]")
    
    pipelines = load_analysis_pipelines()
    if not pipelines:
        console.print("[bold red]Failed to load AI models. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    while True:
        source_file = select_source_file()
        if not source_file:
            console.print("[bold blue]Exiting application. Goodbye![/bold blue]")
            break
        
        process_channel(source_file, pipelines, verbose)

if __name__ == "__main__":
    app()