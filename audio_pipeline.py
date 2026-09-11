# audio_pipeline.py - Single-stream audio processing with VAD + wake word + STT
#
# Owns the microphone stream. Runs wake word detection, VAD, and buffers
# audio for STT. Calls the STT subprocess and hands text to route_request.

import numpy as np
import sounddevice as sd
import torch
import threading
import time as python_time
import os
import subprocess
import sys
import wave
from collections import deque
from enum import Enum

# Import wake word + VAD
from wakeword_detector import WakeWordDetector
from javad.stream import Pipeline as VadPipeline


# ===== Configuration =====
SAMPLE_RATE = 16000
CHUNK_MS = 160
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 2560 samples

# Wake word
WAKEWORD_MODEL = "models/wakeword/jarvis_robust_final/jarvis_robust.onnx"
WAKEWORD_THRESHOLD = 0.4

# VAD
VAD_MODEL = "precise"
VAD_MODE = "gradual"
VAD_THRESHOLD = 0.55
VAD_MIN_SILENCE_CHUNKS = 3   # 480ms silence ends a segment
VAD_MIN_SPEECH_CHUNKS = 1    # at least 160ms of speech

# Follow-up
FOLLOWUP_SILENCE_SECONDS = 5.0
MAX_LISTEN_SECONDS = 15.0

# Temp file
TEMP_DIR = "data/tmp"
TEMP_WAV = os.path.join(TEMP_DIR, "utterance.wav")


class State(Enum):
    SLEEPING = "sleeping"        # Only wake word matters
    LISTENING = "listening"      # Buffering user's speech
    PROCESSING = "processing"    # Transcribing + responding
    FOLLOW_UP = "follow_up"      # Short window for follow-up without wake word


class AudioPipeline:
    def __init__(self, on_transcription, stt_venv_python, stt_service_script):
        """
        Args:
            on_transcription: callback(text) called when we have a transcription
            stt_venv_python: path to venv_stt's python.exe
            stt_service_script: path to stt_service.py
        """
        self.on_transcription = on_transcription
        self.stt_venv_python = stt_venv_python
        self.stt_service_script = stt_service_script

        # State
        self.state = State.SLEEPING
        self.state_lock = threading.Lock()
        self.is_running = False
        self.is_speaking = False  # Set True while TTS is playing

        # Wake word detector
        print("🔊 Loading wake word detector...")
        self.wakeword = WakeWordDetector(
            model_path=WAKEWORD_MODEL,
            threshold=WAKEWORD_THRESHOLD,
        )

        # VAD pipeline
        print("🔊 Loading VAD...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vad = VadPipeline(
            model_name=VAD_MODEL,
            mode=VAD_MODE,
            threshold=VAD_THRESHOLD,
            device=device,
        )
        if hasattr(self.vad, 'to'):
            self.vad.to(device)
        print(f"✅ VAD on {device}")

        # Buffers
        self.audio_buffer = []            # chunks accumulated while LISTENING
        self.consecutive_speech = 0
        self.consecutive_silence = 0
        self.listen_start_time = 0.0

        # Follow-up tracking
        self.followup_silence_start = 0.0

        # Stream
        self.stream = None

        # STT subprocess
        self.stt_proc = None

        # Make temp dir
        os.makedirs(TEMP_DIR, exist_ok=True)

    # ===================== STT Subprocess =====================

    def start_stt_service(self):
        """Launch the STT subprocess and wait for READY."""
        if not os.path.exists(self.stt_venv_python):
            raise FileNotFoundError(f"venv_stt python not found: {self.stt_venv_python}")
        if not os.path.exists(self.stt_service_script):
            raise FileNotFoundError(f"stt_service.py not found: {self.stt_service_script}")

        print(f"🔊 Starting STT service ({self.stt_venv_python})...")
        self.stt_proc = subprocess.Popen(
            [self.stt_venv_python, self.stt_service_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line buffered
            universal_newlines=True,
        )

        # Wait for READY line (up to 90s for first-time model load)
        print("⏳ Waiting for Parakeet to load...")
        start = python_time.time()
        while python_time.time() - start < 90:
            line = self.stt_proc.stdout.readline()
            if not line:
                # Check if it crashed
                if self.stt_proc.poll() is not None:
                    err = self.stt_proc.stderr.read()
                    raise RuntimeError(f"STT service crashed:\n{err}")
                continue
            line = line.strip()
            if line == "READY":
                print(f"✅ STT service ready ({python_time.time() - start:.1f}s)")
                # Drain stderr in a background thread so it doesn't block
                threading.Thread(target=self._drain_stderr, daemon=True).start()
                return
        raise TimeoutError("STT service didn't become ready within 90s")

    def _drain_stderr(self):
        """Read stderr lines from STT service and print them (for debugging)."""
        try:
            for line in self.stt_proc.stderr:
                line = line.rstrip()
                if line:
                    print(f"[stt] {line}")
        except Exception:
            pass

    def transcribe_file(self, path):
        """Send path to STT service, get transcription back."""
        if not self.stt_proc or self.stt_proc.poll() is not None:
            print("⚠️ STT service not running")
            return ""
        try:
            self.stt_proc.stdin.write(path + "\n")
            self.stt_proc.stdin.flush()
            result = self.stt_proc.stdout.readline()
            if not result:
                print("⚠️ STT service returned nothing")
                return ""
            result = result.strip()
            if result.startswith("__ERROR__"):
                print(f"⚠️ STT error: {result}")
                return ""
            return result
        except Exception as e:
            print(f"⚠️ STT communication error: {e}")
            return ""

    def shutdown_stt(self):
        if self.stt_proc and self.stt_proc.poll() is None:
            try:
                self.stt_proc.stdin.write("__EXIT__\n")
                self.stt_proc.stdin.flush()
                self.stt_proc.wait(timeout=5)
            except Exception:
                self.stt_proc.kill()

    # ===================== Audio Stream =====================

    def start(self):
        if self.is_running:
            return
        self.is_running = True

        # Start STT service first
        self.start_stt_service()

        # Then audio stream
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
        )
        self.stream.start()
        print("🎤 Audio pipeline running. Say 'Jarvis' to wake.")

    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.shutdown_stt()
        print("🛑 Audio pipeline stopped.")

    # ===================== Callbacks & State =====================

    def set_speaking(self, speaking):
        """Called by assistant to tell us when TTS is playing."""
        self.is_speaking = speaking

    def _save_buffer_to_wav(self):
        """Save accumulated audio chunks to temp WAV."""
        if not self.audio_buffer:
            return None
        # Concatenate chunks, convert float32 [-1,1] back to int16
        audio = np.concatenate(self.audio_buffer)
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        with wave.open(TEMP_WAV, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return TEMP_WAV

    def _transition_to(self, new_state):
        with self.state_lock:
            if self.state != new_state:
                print(f"🔁 State: {self.state.value} → {new_state.value}")
                self.state = new_state

    def _audio_callback(self, indata, frames, pa_time, status):
        if not self.is_running:
            return

        try:
            # Convert to float32 [-1, 1]
            audio = indata.flatten().astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))

            # If TTS is speaking, ignore everything (don't hear ourselves)
            if self.is_speaking:
                return

            with self.state_lock:
                state = self.state

            # ===== SLEEPING: only wake word matters =====
            if state == State.SLEEPING:
                # Run wake word
                ww_result = self.wakeword.predict(audio)
                if ww_result:
                    print("🎤 Wake word detected!")
                    self.vad.reset() if hasattr(self.vad, 'reset') else None
                    self.audio_buffer = []
                    self.consecutive_speech = 0
                    self.consecutive_silence = 0
                    self.listen_start_time = python_time.time()
                    self._transition_to(State.LISTENING)
                return

            # ===== LISTENING: buffer audio, watch for end-of-speech =====
            if state == State.LISTENING:
                # Run VAD on this chunk
                is_speech = self.vad.detect(audio)

                # Buffer the audio (regardless)
                self.audio_buffer.append(audio.copy())

                # Update consecutive counters
                if is_speech:
                    self.consecutive_speech += 1
                    self.consecutive_silence = 0
                else:
                    self.consecutive_silence += 1

                # Check for end-of-speech
                end_of_speech = (
                    self.consecutive_speech >= VAD_MIN_SPEECH_CHUNKS
                    and self.consecutive_silence >= VAD_MIN_SILENCE_CHUNKS
                )

                # Check for max listen time
                elapsed = python_time.time() - self.listen_start_time
                timeout = elapsed > MAX_LISTEN_SECONDS

                if end_of_speech or timeout:
                    if timeout and self.consecutive_speech == 0:
                        # Nothing was said — abort
                        print("⏱️ Listen timeout with no speech, going back to sleep.")
                        self._transition_to(State.SLEEPING)
                        return

                    # We have speech — process it
                    self._transition_to(State.PROCESSING)
                    threading.Thread(
                        target=self._process_and_respond,
                        daemon=True,
                    ).start()
                return

            # ===== FOLLOW_UP: like LISTENING but no wake word needed =====
            if state == State.FOLLOW_UP:
                is_speech = self.vad.detect(audio)

                if is_speech:
                    # User is speaking — start a new LISTENING cycle
                    print("🎤 Follow-up speech detected")
                    self.vad.reset() if hasattr(self.vad, 'reset') else None
                    self.audio_buffer = [audio.copy()]
                    self.consecutive_speech = 1
                    self.consecutive_silence = 0
                    self.listen_start_time = python_time.time()
                    self._transition_to(State.LISTENING)
                else:
                    # Track how long we've been silent
                    if self.followup_silence_start == 0.0:
                        self.followup_silence_start = python_time.time()
                    elif python_time.time() - self.followup_silence_start > FOLLOWUP_SILENCE_SECONDS:
                        print(f"⏱️ {FOLLOWUP_SILENCE_SECONDS}s of silence, back to sleep.")
                        self._transition_to(State.SLEEPING)
                        self.followup_silence_start = 0.0
                return

        except Exception as e:
            if not hasattr(self, '_error_shown'):
                print(f"⚠️ Audio callback error: {e}")
                import traceback
                traceback.print_exc()
                self._error_shown = True

    def _process_and_respond(self):
        """Save buffer, transcribe, hand text to callback."""
        try:
            wav_path = self._save_buffer_to_wav()
            if not wav_path:
                print("⚠️ No audio to process")
                self._transition_to(State.SLEEPING)
                return

            print(f"📝 Transcribing...")
            text = self.transcribe_file(wav_path)

            if not text:
                print("⚠️ Empty transcription, going back to sleep.")
                self._transition_to(State.SLEEPING)
                return

            print(f"🗣️  You said: {text}")

            # Hand off to the assistant
            try:
                self.on_transcription(text)
            except Exception as e:
                print(f"⚠️ on_transcription error: {e}")

            # After response: enter follow-up mode
            self.followup_silence_start = 0.0
            self._transition_to(State.FOLLOW_UP)

        except Exception as e:
            print(f"⚠️ Process error: {e}")
            import traceback
            traceback.print_exc()
            self._transition_to(State.SLEEPING)