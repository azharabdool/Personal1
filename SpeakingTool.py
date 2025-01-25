from transformers import BertTokenizer, BertForSequenceClassification
import torch
import whisper
import os
from flask import Flask, request, render_template
from pydub.utils import mediainfo

# Initialize Flask app
app = Flask(__name__)

# Whisper model setup
whisper_model = whisper.load_model("base")  # Whisper model setup

# BERT model setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to get audio file duration
def get_audio_duration(audio_path):
    audio_info = mediainfo(audio_path)
    return float(audio_info['duration'])  # Duration in seconds

# Function to transcribe audio with Whisper
def transcribe_audio_with_whisper(audio_path):
    try:
        # Transcribing the audio
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

# Function to grade the transcription using BERT
def grade_response_with_bert(transcription, audio_path):
    # Tokenize the transcription
    inputs = tokenizer(transcription, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get BERT predictions
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Predicted sentiment class (e.g., positive, negative)

    # Fluency based on BERT's sentiment prediction (for simplicity, let's assume it's 0-10 scale)
    fluency_score = 7.0 if predicted_class == 1 else 3.0  # Positive sentiment = higher score, negative = lower score

    # Continue with other grading metrics as before
    words = transcription.split()
    num_words = len(words)
    unique_words = len(set(words))
    lexical_score = (unique_words / num_words) * 10 if num_words > 0 else 0
    grammar_score = 7.0  # Fixed for simplicity

    total_score = round((fluency_score + lexical_score + grammar_score), 2)

    return round(fluency_score, 2), round(lexical_score, 2), round(grammar_score, 2), total_score

# Route for index page (HTML form)
@app.route('/')
def index():
    return render_template('index.html', transcription=None)

# Route to handle file upload and process the audio
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part provided.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Transcribe audio
    transcription = transcribe_audio_with_whisper(file_path)
    if transcription:
        # Grade the transcription using BERT
        fluency, lexical, grammar, total = grade_response_with_bert(transcription, file_path)

        # Display the results
        return render_template('index.html', transcription=transcription, 
                               fluency=fluency, lexical=lexical, 
                               grammar=grammar, total=total)
    else:
        return render_template('index.html', error="Error occurred during transcription.")

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
