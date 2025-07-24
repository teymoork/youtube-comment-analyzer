YouTube Comment Analyzer
This project is an interactive command-line tool 🛠️ for performing advanced, multi-stage analysis on Persian YouTube comments. It processes large JSON data files, applying a series of Hugging Face models to derive rich insights from the text.

✨ Features
Interactive CLI: A user-friendly, menu-driven interface for selecting channels and analysis tasks.

Multi-Stage Analysis Pipeline: Each comment undergoes a four-stage analysis:

Persian Emotion Analysis: Classifies the original comment using a native Persian model.

Translation: Translates the Persian comment into English.

English Emotion Analysis: Classifies the translated text into one of 7 emotions.

Irony Detection: Analyzes the translated text for irony and sarcasm.

Persistent Results: Saves all analysis results to the processed_data/ directory. It automatically skips previously analyzed comments, saving time and computational resources on subsequent runs. ⚡

Configurable Data Source: The input directory for channel data is easily configured in src/config.py.

Robust Dependency Management: Uses Poetry for reproducible dependency management.

🚀 Getting Started
Prerequisites
Python 3.9+

Poetry

Git

Essential build tools. On Debian/Ubuntu, install them with:

Bash

sudo apt update && sudo apt install build-essential cmake pkg-config
Installation
Clone the repository:

Bash

git clone <your-repository-url>
cd youtube_comment_analyzer
Configure Poetry (Optional but Recommended):
To create the virtual environment inside the project folder (.venv), run:

Bash

poetry config virtualenvs.in-project true
Install dependencies:
This command creates the virtual environment and installs all required packages from pyproject.toml.

Bash

poetry install
Note
The first time you run the application, it will download several gigabytes of model data from Hugging Face. This is a one-time process.

Configure Data Source:
Open src/config.py and set the INPUT_DATA_DIR variable to point to your directory of channel JSON files.

Python

# src/config.py
from pathlib import Path

INPUT_DATA_DIR = Path("/path/to/your/channel/data/files/")
💻 Usage
To start the interactive application, run the following command from the project's root directory:

Bash

poetry run python main.py run
This will launch the interactive menu.

Verbose Mode
To see analysis results printed to the console in real-time as they are generated, use the --verbose or -v flag:

Bash

poetry run python main.py run --verbose
