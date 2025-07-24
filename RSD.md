# /home/tkh/repos/hugging_face/youtube_comment_analyzer/RSD.md

# Requirements and Specifications Document (RSD)

## 1. Project Overview

The YouTube Comment Analyzer is a Python-based command-line tool designed to perform multi-faceted linguistic analysis on Persian YouTube comments. It operates interactively, reading channel data from an external source and saving enriched analysis data to a local directory, ensuring that previously processed comments are not re-analyzed.

## 2. Core Requirements

### 2.1. Data Input
- The tool must read channel data from JSON files located in an external directory, specified by the `INPUT_DATA_DIR` variable in `src/config.py`.
- The input JSON files are expected to conform to the `ChannelData` Pydantic schema defined in `src/schemas.py`.

### 2.2. Data Output
- The tool must save its analysis results to a local directory, specified by `PROCESSED_DATA_DIR` in `src/config.py`.
- For an input file named `channel-X.json`, the output file will be named `processed_channel-X.json`.
- The output file will contain the original channel data structure, with an added top-level key: `analysis_results`.
- The `analysis_results` object will be a dictionary where keys are `comment_id`s and values are the corresponding analysis objects.

### 2.3. Persistence
- Before analyzing a comment, the tool must check if a result for that `comment_id` already exists in the loaded `analysis_results` object.
- If a result exists, the comment must be skipped to prevent redundant processing.

## 3. Analysis Pipeline

Each new comment must be processed through a four-stage pipeline:

1.  **Persian Emotion Analysis:**
    -   **Model:** `pars-ai/bert-base-parsbert-uncased-emotion-analysis`
    -   **Input:** Original Persian comment text.
    -   **Output:** A dictionary of scores for all emotion labels.

2.  **Translation (Persian to English):**
    -   **Model:** `Helsinki-NLP/opus-mt-fa-en`
    -   **Input:** Original Persian comment text.
    -   **Output:** The English translation as a string.

3.  **English Emotion Analysis:**
    -   **Model:** `j-hartmann/emotion-english-distilroberta-base`
    -   **Input:** The translated English text.
    -   **Output:** A dictionary of scores for all 7 emotion labels (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).

4.  **Irony Detection:**
    -   **Model:** `cardiffnlp/twitter-roberta-base-irony`
    -   **Input:** The translated English text.
    -   **Output:** A dictionary containing the predicted label (`irony` or `not irony`) and its confidence score.

## 4. Operational Modes (User Interface)

The application must operate as an interactive command-line tool with the following menu structure:

### 4.1. Main Menu
- Lists all `.json` files found in the `INPUT_DATA_DIR`.
- Prompts the user to select a channel file to process or to quit.

### 4.2. Channel Sub-Menu
After a channel is selected, a new menu provides the following options:
- **Batch Analysis:** Prompts the user for the number of videos (`m`) and the number of new comments per video (`n`) to analyze.
- **Single Video Analysis:** Lists all videos in the channel and prompts the user to select one. Analyzes all new comments for the chosen video.
- **Save and Exit:** Saves all new analysis results to the corresponding `processed_*.json` file and returns to the Main Menu.
- **Exit without Saving:** Returns to the Main Menu, discarding any new analysis performed in the current session.

## 5. Technical Specifications

- **Language:** Python 3.9+
- **Dependency Management:** Poetry
- **Key Libraries:**
    - `typer`: For the command-line interface.
    - `rich`: For creating user-friendly menus and formatted output.
    - `transformers`: For interacting with Hugging Face models.
    - `torch`: As the backend for `transformers`.
    - `pydantic`: For data validation and schema enforcement.
    - `loguru`: For application-wide logging.
    - `sentencepiece`: As a required dependency for the translation model.
- **Version Control:** The project repository must include a `.gitignore` file to exclude the `.venv` directory, `__pycache__`, and data directories.