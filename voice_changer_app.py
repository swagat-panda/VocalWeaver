import customtkinter as ctk
import sounddevice as sd
from scipy.io.wavfile import write, read as read_wav  # Import 'read' explicitly
from piper.voice import PiperVoice
from faster_whisper import WhisperModel
import threading
import os
import wave

# --- Configuration ---
MODEL_SIZE = "base.en"  # Whisper model for transcription
VOICES_DIR = "voices"  # Folder where Piper models are stored
AUDIO_FILE = "temp_recording.wav"
TTS_OUTPUT_FILE = "tts_output.wav"
SAMPLE_RATE = 16000
RECORD_SECONDS = 10

# --- Global Placeholders for Models ---
stt_model = None
current_voice_model = None
current_voice_name = None


# --- Core Functions ---
def record_audio():
    """Records audio from the microphone for a fixed duration."""
    print("Recording...")
    recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, recording)
    print("Recording finished.")
    return AUDIO_FILE


def transcribe_audio(file_path):
    """Transcribes the recorded audio file to text using Whisper."""
    print("Transcribing...")
    segments, _ = stt_model.transcribe(file_path, beam_size=5)
    transcribed_text = "".join(segment.text for segment in segments).strip()
    print(f"Transcribed text: {transcribed_text}")
    return transcribed_text


# --- Main Application GUI and Logic ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Offline Voice Changer")
        self.geometry("450x400")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Voice Selection Dropdown ---
        self.voice_label = ctk.CTkLabel(self, text="Select a Voice:", font=("Helvetica", 14))
        self.voice_label.pack(pady=(20, 5), padx=20)

        self.available_voices = self.scan_for_voices()
        voice_names = list(self.available_voices.keys())

        self.voice_variable = ctk.StringVar(value=voice_names[0] if voice_names else "No voices found")
        self.voice_menu = ctk.CTkOptionMenu(self, values=voice_names, variable=self.voice_variable)
        self.voice_menu.pack(pady=5, padx=40, fill="x")

        # --- Main Controls and Status Display ---
        self.status_label = ctk.CTkLabel(self, text="Press the button and speak for 5 seconds.", wraplength=400,
                                         font=("Helvetica", 16))
        self.status_label.pack(pady=20, padx=20)

        self.record_button = ctk.CTkButton(self, text="Record & Speak", command=self.start_processing_thread,
                                           font=("Helvetica", 18))
        self.record_button.pack(pady=20, padx=20, ipady=10)

        if not voice_names:
            self.record_button.configure(state="disabled")
            self.status_label.configure(text=f"Error: No voice models found in the '{VOICES_DIR}' folder.")

        self.textbox = ctk.CTkTextbox(self, height=100, width=380, font=("Helvetica", 12))
        self.textbox.pack(pady=10, padx=10)
        self.textbox.insert("0.0", "Transcribed text will appear here...")
        self.textbox.configure(state="disabled")

    def scan_for_voices(self):
        """Scans the voices directory to find available Piper models."""
        voices = {}
        if not os.path.isdir(VOICES_DIR):
            os.makedirs(VOICES_DIR)
            return {}

        for file in sorted(os.listdir(VOICES_DIR)):
            if file.endswith(".onnx"):
                base_name = file.replace(".onnx", "")
                if os.path.exists(os.path.join(VOICES_DIR, f"{base_name}.onnx.json")):
                    try:
                        parts = base_name.split('-')
                        lang_region = parts[0].split('_')[1]
                        name = parts[1].replace("_", " ").title()
                        quality = parts[2]
                        friendly_name = f"{name} ({lang_region}, {quality})"
                    except IndexError:
                        friendly_name = base_name

                    voices[friendly_name] = {
                        "model": os.path.join(VOICES_DIR, file),
                        "config": os.path.join(VOICES_DIR, f"{base_name}.onnx.json")
                    }
        print(f"Found voices: {list(voices.keys())}")
        return voices

    def speak_text_piper(self, text):
        """Synthesizes text using Piper and plays the audio."""
        global current_voice_model, current_voice_name

        if not text:
            print("No text to speak.")
            return

        selected_voice_friendly_name = self.voice_variable.get()

        if selected_voice_friendly_name != current_voice_name:
            print(f"Loading new voice: {selected_voice_friendly_name}")
            self.status_label.configure(text=f"Loading voice: {selected_voice_friendly_name}...")
            self.update_idletasks()

            voice_files = self.available_voices[selected_voice_friendly_name]
            current_voice_model = PiperVoice.load(voice_files["model"], config_path=voice_files["config"])
            current_voice_name = selected_voice_friendly_name
            print("Voice loaded.")

        self.status_label.configure(text="Synthesizing audio...")
        with wave.open(TTS_OUTPUT_FILE, 'wb') as wav_file:
            current_voice_model.synthesize(text, wav_file)

        self.status_label.configure(text="Speaking...")
        # +++ THIS IS THE CORRECTED LINE +++
        fs, data = read_wav(TTS_OUTPUT_FILE)
        sd.play(data, fs)
        sd.wait()

        if os.path.exists(TTS_OUTPUT_FILE):
            os.remove(TTS_OUTPUT_FILE)

    def process_voice(self):
        """The main workflow: record -> transcribe -> speak."""
        self.record_button.configure(state="disabled", text="Processing...")

        self.status_label.configure(text="Recording...")
        audio_file = record_audio()

        self.status_label.configure(text="Transcribing...")
        transcribed_text = transcribe_audio(audio_file)

        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", transcribed_text or "No speech detected.")
        self.textbox.configure(state="disabled")

        self.speak_text_piper(transcribed_text)

        if os.path.exists(audio_file):
            os.remove(audio_file)

        self.record_button.configure(state="normal", text="Record & Speak")
        self.status_label.configure(text="Press the button and speak for 60 seconds.")

    def start_processing_thread(self):
        """Runs the main process in a separate thread to keep the GUI from freezing."""
        thread = threading.Thread(target=self.process_voice)
        thread.daemon = True
        thread.start()


def main():
    global stt_model
    print("Loading Whisper STT model... (This may take a moment on first run)")
    stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Whisper model loaded successfully.")

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()