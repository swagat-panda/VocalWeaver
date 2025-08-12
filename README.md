# VocalWeaver
VocalWeaver is a real-time, AI-powered voice-changing web service. Speak into your microphone, and VocalWeaver will transcribe your speech and say it back in a different, high-quality synthetic voice, all with low latency.


This project uses a powerful combination of OpenAI's Whisper for transcription and Piper for high-quality, offline Text-to-Speech, all served via a modern FastAPI and WebSocket backend.

Core Features
Real-Time Processing: Speak and hear the transformed voice in seconds.
High-Quality Voices: Utilizes the lightweight and natural-sounding Piper TTS engine.
Accurate Transcription: Powered by a CPU-optimized version of OpenAI's Whisper model.
Web-Based Interface: Accessible from any modern browser on any device on your network.
Fully Offline Models: After initial setup, the AI models run entirely on your machine, requiring no cloud APIs or internet connection.
Selectable Voices: Easily switch between different pre-configured voices (e.g., American/British, Male/Female).
Debug Mode: Includes an optional mode to save intermediate audio files for analysis and debugging.


Technology Stack
Backend: 🐍 Python 3.11
Web Framework: 🚀 FastAPI
Real-Time Communication: 🌐 WebSockets
Web Server: 🦄 Uvicorn
Speech-to-Text (STT): 🤫 faster-whisper (CPU-optimized Whisper)
Text-to-Speech (TTS): 🎤 piper-tts
Audio Processing: 🎶 pydub & scipy
External Dependency: 🎬 FFmpeg
Setup and Installation
Follow these steps to get VocalWeaver running on your local machine.

1. Prerequisites
Python: This project requires Python 3.10 or 3.11.
FFmpeg: The pydub library requires FFmpeg for audio format conversion. You must install it on your system.
Windows (easiest way): Open PowerShell and run winget install "FFmpeg (Essentials Build)".
macOS: Run brew install ffmpeg.
Linux (Debian/Ubuntu): Run sudo apt update && sudo apt install ffmpeg.
After installation, close and reopen your terminal. Verify it works by typing ffmpeg -version.
2. Clone the Repository
git clone https://github.com/your-username/VocalWeaver.git
cd VocalWeaver
3. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
4. Install Dependencies
Install all required Python libraries from the requirements.txt file.

pip install -r requirements.txt
5. Download TTS Voice Models
The high-quality voices are provided by Piper. You need to download them manually.

Navigate to the Piper Voices Collection on Hugging Face.
Download the .onnx and .onnx.json files for each voice you want. The recommended set is:
en_US/ryan/medium (American Male)
en_US/ljspeech/medium (American Female)
en_GB/alan/low (British Male)
en_GB/southern_english_female/medium (British Female)
Place all downloaded model files (e.g., en_US-ryan-medium.onnx and en_US-ryan-medium.onnx.json) inside the voices/ directory in the project root.
Running the Application
Start the Server:
Run the Uvicorn server from the project's root directory.

uvicorn main:app --reload
The server will start, and you will see logs indicating that the Whisper and Piper models are being loaded into memory. This may take a moment on the first run.

Access the UI:
Open your web browser and navigate to:
http://127.0.0.1:8000

Grant Microphone Access:
Your browser will prompt you for permission to use your microphone. You must Allow it for the application to work.

Configuration
Debug Mode
The main.py file contains a debug flag:

# Set this to True to save intermediate audio files.
DEBUG_SAVE_FILES = True
When DEBUG_SAVE_FILES is True, the server will save the received, converted, and synthesized audio files into the debug_audio/ folder. This is excellent for troubleshooting.
When set to False, the application runs entirely in memory for better performance.
Project Structure
VocalWeaver/
├── static/
│   └── index.html          # The frontend user interface
├── voices/
│   └── ...                 # Place your downloaded Piper TTS models here
├── debug_audio/            # Auto-created in debug mode for audio files
├── main.py                 # The FastAPI server, WebSocket logic, and AI pipeline
├── requirements.txt        # Project dependencies
└── README.md               # This file
How It Works
The application follows a simple, real-time pipeline:

Client (Browser): The user records audio using the MediaRecorder API, which creates a WebM audio blob.
WebSocket Send: The audio blob and selected voice are sent to the FastAPI server over a WebSocket connection.
Server - Conversion: The server receives the WebM data and uses pydub (with FFmpeg) to convert it to a standard 16kHz mono WAV format.
Server - Transcription: The WAV data is passed to the faster-whisper model, which transcribes the speech to text.
Server - Synthesis: The transcribed text is given to the selected piper-tts voice model, which generates new audio data in WAV format.
WebSocket Receive: The server sends the final audio data and the transcribed text back to the client.
Client (Browser): The JavaScript code receives the new audio, creates a playable blob, and plays it automatically.