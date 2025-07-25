# /home/tkh/repos/hugging_face/youtube_comment_analyzer/README.md

# YouTube Comment Analyzer

This project is an interactive command-line tool for performing advanced, multi-stage analysis on Persian YouTube comments. It is designed to process large JSON data files, applying a series of Hugging Face models to derive rich insights from comment text.

The application is built to be robust against network issues by loading all AI models from local storage.

## Features

- **Interactive CLI:** A user-friendly, menu-driven interface for selecting channels and analysis tasks.
- **Local Model Execution:** All Hugging Face models are loaded from the local filesystem, bypassing network connectivity issues and ensuring consistent, offline operation.
- **Multi-Stage Analysis Pipeline:** Each comment undergoes a four-stage analysis:
    1.  **Persian Sentiment Analysis:** Classifies the original comment text.
    2.  **Translation:** Translates the Persian comment into English.
    3.  **English Emotion Analysis:** Classifies the translated text into a standard set of 7 emotions.
    4.  **Irony Detection:** Analyzes the translated text for irony and sarcasm.
- **Decoupled & Persistent Results:** The tool saves all analysis results to separate `analysis_...json` files. It automatically detects previously analyzed comments and skips them, saving time and computational resources on every run.
- **Robust Data Handling:** Uses Pydantic schemas to safely parse and validate input data, gracefully handling missing or incomplete sections (e.g., videos without comments).
- **Comprehensive Logging:** Logs all operations to both the console and a persistent `app_log.txt` file for easy debugging.

## Setup

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/docs/#installation) for package management.
- Git for version control.
- Essential build tools for compiling certain Python packages. On Debian/Ubuntu, install them with:
  ```bash
  sudo apt update && sudo apt install build-essential cmake pkg-config
Use code with caution.
Markdown
Installation
Clone the repository:
Generated bash
git clone <your-repository-url>
cd youtube_comment_analyzer
Use code with caution.
Bash
Configure Poetry (Recommended):
To ensure the virtual environment is created inside the project folder (as .venv/), run this command once:
Generated bash
poetry config virtualenvs.in-project true
Use code with caution.
Bash
Install dependencies:
This command will create a .venv virtual environment and install all required packages.
Generated bash
poetry install
Use code with caution.
Bash
Download AI Models:
Due to network restrictions, the models must be downloaded manually from an unrestricted network (e.g., using a VPN or a different computer) and placed in a ~/huggingface_models directory in your home folder. See the RSD.md for the exact file structure and download links.
Configure Data Source:
Open the src/config.py file and ensure the INPUT_DATA_DIR variable points to the directory containing your channel JSON files.
Usage
To start the application, run the following command from the project's root directory:
Generated bash
poetry run python main.py
Use code with caution.
Bash
This will launch the interactive menu.
Verbose Mode
To see the analysis results printed to the console in real-time as they are generated, use the --verbose or -v flag:
Generated bash
poetry run python main.py --verbose
