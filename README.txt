# Audio Transcription and Grading Tool
# Azhar Abdool
#


This project is a fully functional web-based tool that transcribes audio files and grades the transcription based on fluency, lexical diversity, and grammar using Whisper and BERT models.

---

## Features

- **Audio Transcription**: Uses OpenAI's Whisper model to transcribe uploaded audio files.
- **Grading System**: Employs a BERT model for sentiment analysis and scores the transcription based on:
  - **Fluency**: Derived from sentiment analysis.
  - **Lexical Diversity**: The ratio of unique words to the total number of words.
  - **Grammar**: A static score for demonstration purposes.
- **Interactive Web Interface**: A simple, user-friendly interface for uploading audio files and viewing results.


---

## Setup Instructions

### Prerequisites

1. Python 3.8 or later
2. Virtual environment (optional, but recommended)
3. Required Python libraries: `transformers`, `torch`, `flask`, `whisper`, `pydub`

### Installation

1. Clone the repository:
https://github.com/azharabdool/Personal1.git
2. Create a virtual environment and activate it:
3. Install the required dependencies:
Pip package manager
FFmpeg (required for processing audio files)
4.Running the Application
Start the Flask server:
python app.py
Open your browser and navigate to:
http://127.0.0.1:5000/

Project Architecture
Backend
Whisper: Used for audio transcription.
BERT: Provides a sentiment analysis-based fluency score and grading system.
Frontend
Flask Templates: Basic HTML templates to display the web interface.
Grading System
Fluency Score:
Based on the sentiment of the transcription (Positive → High, Negative → Low).
Lexical Diversity:
Ratio of unique words to total words, scaled to a score out of 10.
Grammar:
Static score for simplicity (customizable in future iterations).


Contact
For any questions or feedback, feel free to contact:

Name: Azhar Abdool
Email: azharabdool786@gmail.com
GitHub: https://github.com/azharabdool

