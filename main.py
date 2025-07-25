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

# Import our own modules
from src.logger_config import app_logger
from src.config import (
    INPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_COMMENTS_PER_VIDEO,
)
from src.schemas import ChannelData, AnalysisResult, StoredComment
from src.data_utils import (
    load_app_data,
    save_app_data,
    update_data_from_source,
)
from src.analysis import (
    load_analysis_pipelines,
    analyze_persian_emotion,
    translate_text,
    analyze_english_emotion,
    analyze_irony,
)
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

def run_analysis_on_comment(
    comment: StoredComment,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
) -> None:
    """Performs a multi-stage analysis and attaches the result to the comment object."""
    if not comment.text_original or not comment.text_original.strip():
        app_logger.warning(f"Skipping comment {comment.comment_id} due to empty text.")
        return

    comment_text = comment.text_original
    MAX_COMMENT_CHARS = 500
    if len(comment_text) > MAX_COMMENT_CHARS:
        app_logger.warning(f"Comment {comment.comment_id} is too long ({len(comment_text)} chars). Truncating to {MAX_COMMENT_CHARS}.")
        comment_text = comment_text[:MAX_COMMENT_CHARS]

    app_logger.info(f"Analyzing comment ID: {comment.comment_id}")
    
    persian_emotion_pipeline, translation_pipeline, english_emotion_pipeline, irony_pipeline = pipelines
    
    persian_sentiment_result = analyze_persian_emotion(comment_text, persian_emotion_pipeline)
    translated_text = translate_text(comment_text, translation_pipeline)
    
    english_emotion_result = None
    irony_result = None
    if translated_text:
        english_emotion_result = analyze_english_emotion(translated_text, english_emotion_pipeline)
        irony_result = analyze_irony(translated_text, irony_pipeline)
    else:
        app_logger.warning(f"Translation failed for comment {comment.comment_id}. Skipping English analyses.")

    analysis_result = AnalysisResult(
        persian_sentiment=persian_sentiment_result,
        english_translation=translated_text,
        english_emotions=english_emotion_result,
        english_irony=irony_result,
        analyzed_at=datetime.now(timezone.utc)
    )
    
    # Attach the result directly to the comment object
    comment.analysis = analysis_result

    if verbose:
        console.print(f"\n[bold]Comment ID:[/] {comment.comment_id}")
        console.print(f"[bold]Original Text:[/] '{comment_text[:100]}...'")
        console.print(f"[bold magenta]Persian Sentiment:[/] {analysis_result.persian_sentiment}")
        console.print("---")
        console.print(f"[bold]Translated Text:[/] '{analysis_result.english_translation[:100] if analysis_result.english_translation else 'N/A'}'")
        console.print(f"[bold cyan]English Emotions:[/] {analysis_result.english_emotions}")
        console.print(f"[bold yellow]Irony Detection:[/] {analysis_result.english_irony}")

def handle_batch_analysis(
    channel_data: ChannelData,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
):
    """Handler for analyzing n comments from m videos with detailed progress."""
    num_comments_per_video = IntPrompt.ask(
        f"How many new comments to analyze per video?",
        default=DEFAULT_COMMENTS_PER_VIDEO
    )
    num_videos_to_process = IntPrompt.ask(
        "How many videos to process?",
        default=len(channel_data.videos)
    )
    
    processed_count_total = 0
    videos_to_process = list(channel_data.videos.values())[:num_videos_to_process]

    for video_index, video_data in enumerate(videos_to_process):
        console.print(
            f"\n[bold]Processing Video {video_index + 1}/{len(videos_to_process)}:[/] "
            f"{video_data.video_metadata.title} ({video_data.video_metadata.video_id})"
        )
        
        # Find comments in this video that need analysis
        comments_to_analyze = [c for c in video_data.comments.values() if c.analysis is None]
        
        # Sort by date to process newest first
        sorted_comments = sorted(comments_to_analyze, key=lambda c: c.published_at, reverse=True)
        
        # Take the requested number of comments
        comments_for_this_run = sorted_comments[:num_comments_per_video]
        total_to_analyze_in_video = len(comments_for_this_run)

        if total_to_analyze_in_video == 0:
            console.print("[yellow]No new comments to analyze in this video.[/yellow]")
            continue

        for comment_index, comment in enumerate(comments_for_this_run):
            console.print(f"  Analyzing comment {comment_index + 1}/{total_to_analyze_in_video}...")
            run_analysis_on_comment(comment, pipelines, verbose)
            processed_count_total += 1
    
    console.print(f"\n[bold green]Batch analysis complete. Analyzed {processed_count_total} new comments.[/bold green]")

def handle_video_analysis(
    channel_data: ChannelData,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
):
    """Handler for analyzing all new comments in a specific video."""
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

    comments_to_analyze = [c for c in selected_video_data.comments.values() if c.analysis is None]
    
    if not comments_to_analyze:
        console.print("\n[bold green]All comments in this video have already been analyzed.[/bold green]")
        return

    processed_count = 0
    for comment in comments_to_analyze:
        run_analysis_on_comment(comment, pipelines, verbose)
        processed_count += 1
    
    console.print(f"\n[bold green]Video analysis complete. Analyzed {processed_count} new comments.[/bold green]")

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
        console.print("  [green]b[/green]: Back to Main Menu")

        choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "b"], default="b")

        if choice == '1':
            console.print(f"\n[bold]Updating from source file: {source_file.name}...[/bold]")
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_dict = json.load(f)
                channel_data = update_data_from_source(channel_data, source_dict)
                save_app_data(app_data_path, channel_data)
            except (json.JSONDecodeError, ValidationError, FileNotFoundError) as e:
                console.print(f"[bold red]Error during update: {e}[/bold red]")
        
        elif choice in ['2', '3']:
            if not channel_data:
                console.print("\n[bold red]Cannot analyze. No data loaded. Please run 'Update from Source' first.[/bold red]")
                continue
            
            if choice == '2':
                handle_batch_analysis(channel_data, pipelines, verbose)
            elif choice == '3':
                handle_video_analysis(channel_data, pipelines, verbose)
            
            save_app_data(app_data_path, channel_data)

        elif choice == '4':
            display_channel_stats(channel_data)

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