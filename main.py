import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from scipy.io.wavfile import read as read_wav
from pydub import AudioSegment
import numpy as np
import io
import os
import base64
import json
import wave

# --- Configuration & Global Variables ---
MODEL_SIZE = "base.en"
VOICES_DIR = "voices"
SAMPLE_RATE = 16000
stt_model = None
tts_voices = {}
available_voices_info = {}

# --- DEBUGGING FLAG ---
DEBUG_SAVE_FILES = True
DEBUG_FOLDER = "debug_audio"

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Model Loading (on server startup) ---
@app.on_event("startup")
async def startup_event():
    global stt_model, tts_voices, available_voices_info
    if DEBUG_SAVE_FILES:
        os.makedirs(DEBUG_FOLDER, exist_ok=True)
        print(f"Debug mode is ON. Saving files to '{DEBUG_FOLDER}/'")
    print("Loading Whisper STT model...")
    stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Whisper model loaded.")
    print("Scanning for Piper TTS voices...")
    if not os.path.isdir(VOICES_DIR):
        raise RuntimeError(f"The voices directory '{VOICES_DIR}' was not found.")
    for file in sorted(os.listdir(VOICES_DIR)):
        if file.endswith(".onnx"):
            base_name = file.replace(".onnx", "")
            config_path = os.path.join(VOICES_DIR, f"{base_name}.onnx.json")
            if os.path.exists(config_path):
                try:
                    parts = base_name.split('-')
                    lang_region = parts[0].split('_')[1]
                    name = parts[1].replace("_", " ").title()
                    quality = parts[2]
                    friendly_name = f"{name} ({lang_region}, {quality})"
                except IndexError: friendly_name = base_name
                print(f"Loading voice: {friendly_name}")
                model_path = os.path.join(VOICES_DIR, file)
                voice = PiperVoice.load(model_path, config_path=config_path)
                tts_voices[friendly_name] = voice
                available_voices_info[friendly_name] = base_name
    print(f"Loaded {len(tts_voices)} TTS voices successfully.")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

def transcribe_audio_stream(audio_bytes: bytes) -> str:
    if DEBUG_SAVE_FILES:
        with open(os.path.join(DEBUG_FOLDER, "received_audio.webm"), "wb") as f: f.write(audio_bytes)
    try:
        webm_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        webm_audio = webm_audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        wav_buffer = io.BytesIO()
        webm_audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        if DEBUG_SAVE_FILES:
            with open(os.path.join(DEBUG_FOLDER, "converted_audio.wav"), "wb") as f: f.write(wav_buffer.getvalue())
            wav_buffer.seek(0)
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        return ""
    print("Transcribing audio...")
    samplerate, data = read_wav(wav_buffer)
    audio_float32 = data.astype(np.float32) / np.iinfo(data.dtype).max
    segments, _ = stt_model.transcribe(audio_float32, beam_size=5)
    transcribed_text = "".join(segment.text for segment in segments).strip()
    print(f"Transcribed text: {transcribed_text}")
    return transcribed_text

def synthesize_speech_stream(text: str, voice_name: str) -> bytes:
    """Synthesizes speech using the correct in-memory wave object."""
    print(f"Synthesizing speech with voice: {voice_name}")
    if voice_name not in tts_voices:
        raise ValueError("Selected voice not found.")
    
    voice = tts_voices[voice_name]
    
    # 1. Create an in-memory binary buffer.
    audio_stream = io.BytesIO()
    
    # 2. Use wave.open to wrap the buffer. This creates the object that
    #    piper-tts expects, which will write its WAV data into our buffer.
    with wave.open(audio_stream, 'wb') as wav_file:
        voice.synthesize(text, wav_file)
    
    if DEBUG_SAVE_FILES:
        with open(os.path.join(DEBUG_FOLDER, "synthesized_output.wav"), "wb") as f:
            f.write(audio_stream.getvalue())
    
    audio_stream.seek(0)
    return audio_stream.read()

# --- WebSocket Endpoint  ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")
    await websocket.send_json({"type": "voices", "data": list(available_voices_info.keys())})
    try:
        while True:
            message = await websocket.receive_json()
            voice = message.get("voice")
            audio_b64 = message.get("audio")
            if not voice or not audio_b64:
                continue
            await websocket.send_json({"type": "status", "message": "Received audio. Transcribing..."})
            audio_bytes = base64.b64decode(audio_b64)
            transcribed_text = transcribe_audio_stream(audio_bytes)
            if transcribed_text:
                await websocket.send_json({"type": "status", "message": "Synthesizing new voice..."})
                synthesized_bytes = synthesize_speech_stream(transcribed_text, voice)
                synthesized_b64 = base64.b64encode(synthesized_bytes).decode('utf-8')
                await websocket.send_json({"type": "result", "text": transcribed_text, "audio": synthesized_b64})
            else:
                 await websocket.send_json({"type": "result", "text": "No speech detected.", "audio": None})
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason=f"An internal error occurred: {e}")

# --- To run the server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)