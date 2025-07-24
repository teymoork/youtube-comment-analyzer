# /home/tkh/repos/hugging_face/youtube_comment_analyzer/main.py
import typer
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from pydantic import ValidationError
import time

# Import our own modules
from src.logger_config import app_logger
from src.config import (
    INPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_COMMENTS_PER_VIDEO,
)
from src.schemas import ChannelData
from src.data_utils import (
    load_channel_data,
    load_or_initialize_results,
    save_analysis_results,
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

def select_channel_file() -> Optional[Path]:
    """Displays a menu of channel files for the user to choose from."""
    console.print("\n[bold cyan]Select a channel to analyze:[/bold cyan]")
    
    try:
        channel_files = sorted([f for f in INPUT_DATA_DIR.glob('*.json') if f.is_file()])
    except FileNotFoundError:
        console.print(f"[bold red]Error: Input directory not found at '{INPUT_DATA_DIR}'[/bold red]")
        return None

    if not channel_files:
        console.print(f"[yellow]No channel JSON files found in '{INPUT_DATA_DIR}'. Please add some.[/yellow]")
        return None

    for i, file_path in enumerate(channel_files):
        console.print(f"  [green]{i + 1}[/green]: {file_path.name}")
    
    console.print("  [green]q[/green]: Quit")

    choice = Prompt.ask("Enter your choice", default="q")

    if choice.lower() == 'q':
        return None
    
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(channel_files):
            return channel_files[choice_index]
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")
            return select_channel_file()
    except ValueError:
        console.print("[red]Invalid input. Please enter a number or 'q'.[/red]")
        return select_channel_file()

def run_analysis_on_comment(
    comment_id: str,
    comment_text: str,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
) -> Optional[Dict[str, Any]]:
    """Performs a multi-stage analysis and includes the original text in the result."""
    persian_emotion_pipeline, translation_pipeline, english_emotion_pipeline, irony_pipeline = pipelines
    
    if not comment_text or not comment_text.strip():
        app_logger.warning(f"Skipping comment {comment_id} due to empty text.")
        return None

    app_logger.info(f"Analyzing comment ID: {comment_id}")
    
    persian_emotion_result = analyze_persian_emotion(comment_text, persian_emotion_pipeline)
    translated_text = translate_text(comment_text, translation_pipeline)
    
    english_emotion_result = None
    irony_result = None
    if translated_text:
        english_emotion_result = analyze_english_emotion(translated_text, english_emotion_pipeline)
        irony_result = analyze_irony(translated_text, irony_pipeline)
    else:
        app_logger.warning(f"Translation failed for comment {comment_id}. Skipping English analyses.")

    analysis_data = {
        "original_text": comment_text,
        "persian_emotion": persian_emotion_result,
        "translated_text": translated_text,
        "english_emotion": english_emotion_result,
        "irony": irony_result,
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    if verbose:
        console.print(f"\n[bold]Comment ID:[/] {comment_id}")
        console.print(f"[bold]Original Text:[/] '{analysis_data['original_text'][:100]}...'")
        console.print(f"[bold magenta]Persian Emotion:[/] {analysis_data['persian_emotion']}")
        console.print("---")
        console.print(f"[bold]Translated Text:[/] '{analysis_data['translated_text'][:100] if analysis_data['translated_text'] else 'N/A'}'")
        console.print(f"[bold cyan]English Emotion:[/] {analysis_data['english_emotion']}")
        console.print(f"[bold yellow]Irony Detection:[/] {analysis_data['irony']}")

    return analysis_data

def handle_batch_analysis(
    channel_data: ChannelData,
    analysis_results: Dict[str, Any],
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
):
    """Handler for analyzing n comments from m videos."""
    num_comments_per_video = IntPrompt.ask(
        f"How many new comments to analyze per video?",
        default=DEFAULT_COMMENTS_PER_VIDEO
    )
    num_videos = IntPrompt.ask(
        "How many videos to process?",
        default=len(channel_data.videos)
    )
    
    processed_count_total = 0
    videos_processed = 0

    for video_id, video_data in channel_data.videos.items():
        if videos_processed >= num_videos:
            break
        
        console.print(f"\n[bold]Processing Video:[/] {video_data.video_metadata.title} ({video_id})")
        processed_count_video = 0
        
        sorted_comments = sorted(
            video_data.comments.values(),
            key=lambda c: c.published_at,
            reverse=True
        )

        for comment in sorted_comments:
            if processed_count_video >= num_comments_per_video:
                break
            
            # Updated persistence check
            if video_id not in analysis_results or comment.comment_id not in analysis_results[video_id]:
                result = run_analysis_on_comment(
                    comment.comment_id,
                    comment.text_original,
                    pipelines,
                    verbose
                )
                if result:
                    # Updated result storage
                    if video_id not in analysis_results:
                        analysis_results[video_id] = {}
                    analysis_results[video_id][comment.comment_id] = result
                    processed_count_video += 1
                    processed_count_total += 1
        
        if processed_count_video == 0:
            console.print("[yellow]No new comments to analyze in this video.[/yellow]")

        videos_processed += 1
    
    console.print(f"\n[bold green]Batch analysis complete. Analyzed {processed_count_total} new comments across {videos_processed} videos.[/bold green]")
    return analysis_results

def handle_video_analysis(
    channel_data: ChannelData,
    analysis_results: Dict[str, Any],
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
        video_id = selected_video_data.video_metadata.video_id
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/red]")
        return analysis_results

    processed_count = 0
    for comment_id, comment in selected_video_data.comments.items():
        # Updated persistence check
        if video_id not in analysis_results or comment_id not in analysis_results[video_id]:
            result = run_analysis_on_comment(
                comment_id,
                comment.text_original,
                pipelines,
                verbose
            )
            if result:
                # Updated result storage
                if video_id not in analysis_results:
                    analysis_results[video_id] = {}
                analysis_results[video_id][comment_id] = result
                processed_count += 1
    
    console.print(f"\n[bold green]Video analysis complete. Analyzed {processed_count} new comments.[/bold green]")
    return analysis_results

def process_channel(
    channel_file: Path,
    pipelines: Tuple[Pipeline, Pipeline, Pipeline, Pipeline],
    verbose: bool
):
    """Main logic loop for a single selected channel."""
    raw_data = load_channel_data(channel_file)
    if not raw_data:
        return

    try:
        channel_data = ChannelData.model_validate(raw_data)
        console.print(f"[bold green]Successfully validated data for channel: {channel_data.channel_metadata.title}[/bold green]")
    except ValidationError as e:
        console.print(f"[bold red]Data validation error for {channel_file.name}: {e}[/bold red]")
        return

    # Updated output filename
    output_filename = f"analysis_{channel_file.name}"
    output_path = PROCESSED_DATA_DIR / output_filename
    
    # Updated loading function call
    analysis_results = load_or_initialize_results(output_path)
    
    while True:
        console.print("\n[bold cyan]Analysis Options:[/bold cyan]")
        console.print(f"  [green]1[/green]: Analyze new comments (batch mode)")
        console.print(f"  [green]2[/green]: Analyze all new comments from a specific video")
        console.print(f"  [green]s[/green]: Save results and go back to main menu")
        console.print(f"  [green]b[/green]: Go back without saving")

        choice = Prompt.ask("Enter your choice", default="s")

        if choice == '1':
            analysis_results = handle_batch_analysis(channel_data, analysis_results, pipelines, verbose)
        elif choice == '2':
            analysis_results = handle_video_analysis(channel_data, analysis_results, pipelines, verbose)
        elif choice.lower() == 's':
            # Updated saving function call
            save_analysis_results(output_path, analysis_results)
            break
        elif choice.lower() == 'b':
            console.print("[yellow]Returning to main menu without saving changes.[/yellow]")
            break
        else:
            console.print("[red]Invalid choice.[/red]")

@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print analysis results to the screen.")
):
    """
    Starts the interactive YouTube Comment Analyzer tool.
    """
    console.print("[bold blue]--- YouTube Comment Analyzer ---[/bold blue]")
    
    pipelines = load_analysis_pipelines()
    if not pipelines:
        console.print("[bold red]Failed to load AI models. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    while True:
        channel_file = select_channel_file()
        if not channel_file:
            console.print("[bold blue]Exiting application. Goodbye![/bold blue]")
            break
        
        process_channel(channel_file, pipelines, verbose)

if __name__ == "__main__":
    app()