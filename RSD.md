# /home/tkh/repos/hugging_face/youtube_comment_analyzer/RSD.md

# Requirements and Specifications Document (RSD)

## 1. Project Overview

The YouTube Comment Analyzer is a Python-based command-line tool designed to perform multi-faceted linguistic analysis on Persian YouTube comments. It operates interactively, reading channel data from an external source and saving enriched analysis data to a separate, local directory. The system is designed to be robust against network failures by loading all AI models from the local filesystem and to be efficient by never re-analyzing previously processed comments.

## 2. Core Requirements

### 2.1. Data Input
- The tool must read channel data from JSON files located in an external directory, specified by the `INPUT_DATA_DIR` variable in `src/config.py`.
- The input JSON files are expected to conform to the `ChannelData` Pydantic schema defined in `src/schemas.py`. This schema is designed to be resilient to missing `comments` sections in video data.

### 2.2. Data Output
- The tool must save its analysis results to a local directory, specified by `PROCESSED_DATA_DIR` in `src/config.py`.
- For an input file named `channel-X.json`, the output file will be a separate file named `analysis_channel-X.json`.
- The output file will not contain the original channel data, only the analysis results.

### 2.3. Persistence
- Before analyzing a comment, the tool must load the corresponding `analysis_...json` file if it exists.
- It must check if a result for a given `comment_id` within a specific `video_id` already exists in the loaded data.
- If a result exists, the comment must be skipped to prevent redundant processing.

## 3. Data Structures

### 3.1. Input Data Structure (`channel-X.json`)
The input is a single JSON object representing all data for a channel, validated by the `ChannelData` Pydantic model. A simplified structure is:
```json
{
    "channel_metadata": { ... },
    "videos": {
        "video_id_1": {
            "video_metadata": { ... },
            "comments": {
                "comment_id_1": { ... },
                "comment_id_2": { ... }
            }
        }
    }
}
Use code with caution.
Markdown
3.2. Output Data Structure (analysis_channel-X.json)
The output is a single JSON object containing only the analysis results.
The top-level keys are video_ids.
Each video_id object contains the video's publication date and a comments object.
The comments object contains comment_id keys, with the analysis object as the value.
Generated json
{
    "video_id_1": {
        "video_published_at": "YYYY-MM-DDTHH:MM:SSZ",
        "comments": {
            "comment_id_1": {
                "original_text": "...",
                "persian_emotion": { ... },
                "translated_text": "...",
                "english_emotion": { ... },
                "irony": { ... },
                "analyzed_at": "..."
            }
        }
    }
}
Use code with caution.
Json
4. Analysis Pipeline
Each new comment must be processed through a four-stage pipeline using models loaded from local disk paths defined in src/config.py.
Persian Sentiment Analysis:
Model: HooshvareLab/bert-fa-base-uncased-clf-persiannews
Input: Original Persian comment text (truncated if necessary).
Output: A dictionary of scores for sentiment labels.
Translation (Persian to English):
Model: persiannlp/mt5-base-parsinlu-translation-fa-en
Input: Original Persian comment text.
Output: The English translation as a string.
English Emotion Analysis:
Model: j-hartmann/emotion-english-distilroberta-base
Input: The translated English text.
Output: A dictionary of scores for 7 emotion labels.
Irony Detection:
Model: cardiffnlp/twitter-roberta-base-irony
Input: The translated English text.
Output: A dictionary containing the predicted label and its confidence score.
5. Operational Modes (User Interface)
The application operates as an interactive CLI with the following menu structure:
5.1. Main Menu
Lists all .json files found in the INPUT_DATA_DIR.
Prompts the user to select a channel file to process or to quit.
5.2. Channel Sub-Menu
Batch Analysis: Prompts for the number of videos (m) and new comments per video (n) to analyze. Provides detailed progress indicators during processing.
Single Video Analysis: Lists all videos and prompts the user to select one to analyze all its new comments.
Save and Exit: Saves results to the analysis_...json file and returns to the Main Menu.
Exit without Saving: Returns to the Main Menu, discarding new analysis.
6. Technical Specifications
Language: Python 3.9+
Dependency Management: Poetry
Key Libraries:
typer, rich: For the CLI and user interface.
transformers, torch, accelerate: For interacting with Hugging Face models.
pydantic: For data validation.
loguru: For console and file logging.
sentencepiece, protobuf==3.20.3: Required dependencies for specific models.
Version Control: Git, with a .gitignore file to exclude generated files and virtual environments.
