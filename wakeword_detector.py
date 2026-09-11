# wakeword_detector.py - Pure inference, no audio stream, peak detection
#
# The audio stream is owned by audio_pipeline.py. This module only does:
#   - load the ONNX model
#   - provide push_audio(chunk) that rolls an internal 1-second buffer
#     and returns True when the wake word is detected
#
# No sounddevice. No threads. Just inference.

import numpy as np
import time as python_time
import os
from collections import deque

# Suppress ONNX Runtime warnings BEFORE importing onnxruntime
import onnxruntime as ort
ort.set_default_logger_severity(3)  # 3 = ERROR only


class WakeWordDetector:
    def __init__(self,
                 model_path="models/wakeword/jarvis_robust_final/jarvis_robust.onnx",
                 threshold=0.30,
                 sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.cooldown_seconds = 2.0
        self.cooldown_until = 0.0
        self.samples_needed = 16000  # 1 second at 16 kHz

        # Rolling buffer for the 1-second window
        self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)
        self.buffer_filled = 0

        # Confidence history for peak detection
        self.confidence_history = deque(maxlen=3)

        self.session = None
        self.input_name = None
        self.output_name = None

        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return

            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3

            providers = []
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape

            print(f"✅ ONNX model loaded: {model_path}")
            print(f"   Input shape: {input_shape}")
            print(f"   Output shape: {output_shape}")
            print(f"   Providers: {self.session.get_providers()}")
            print(f"   Threshold: {self.threshold}")

            # [batch, 1, 16000] → samples = 16000
            if isinstance(input_shape[2], int):
                self.samples_needed = input_shape[2]
                self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)

        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            self.session = None

    def _predict(self, audio_chunk):
        """Run inference on a 1-second chunk. Returns confidence in [0,1]."""
        if self.session is None:
            return 0.0
        try:
            if len(audio_chunk) < self.samples_needed:
                audio_chunk = np.pad(
                    audio_chunk,
                    (0, self.samples_needed - len(audio_chunk))
                )
            elif len(audio_chunk) > self.samples_needed:
                audio_chunk = audio_chunk[:self.samples_needed]

            # [batch, channels, samples]
            audio_input = audio_chunk.reshape(1, 1, self.samples_needed).astype(np.float32)

            outputs = self.session.run(
                [self.output_name],
                {self.input_name: audio_input},
            )
            confidence = float(np.array(outputs[0]).flatten()[0])
            return max(0.0, min(1.0, confidence))
        except Exception:
            return 0.0

    def push_audio(self, chunk):
        """
        Feed a small audio chunk (e.g. 160 ms = 2560 samples, float32 in [-1, 1]).

        Rolls the internal 1-second buffer and runs inference when the buffer
        is full. Returns True if the wake word was detected on this chunk.

        Peak detection strategy:
        - Track the last few confidence scores.
        - Trigger when current > threshold AND current >= previous
          (i.e. we just crossed the peak).
        This handles the case where "jarvis what's the weather" is said in
        one breath — the peak fires as soon as the buffer contains "jarvis".
        """
        if self.session is None:
            return False

        n = len(chunk)
        if n >= self.samples_needed:
            self.audio_buffer = chunk[-self.samples_needed:].copy()
            self.buffer_filled = self.samples_needed
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -n)
            self.audio_buffer[-n:] = chunk
            self.buffer_filled = min(self.samples_needed, self.buffer_filled + n)

        if self.buffer_filled < self.samples_needed:
            return False

        now = python_time.time()
        if now < self.cooldown_until:
            return False

        confidence = self._predict(self.audio_buffer.copy())

        self.confidence_history.append(confidence)
        if len(self.confidence_history) < 2:
            return False

        prev = self.confidence_history[-2]
        detected = confidence > self.threshold and confidence >= prev

        if detected:
            print(f"   [wake] confidence={confidence:.3f} (prev={prev:.3f})")
            self.cooldown_until = now + self.cooldown_seconds
            # Flush buffer so old audio doesn't re-trigger
            self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)
            self.buffer_filled = 0
            self.confidence_history.clear()
            return True

        return False

    def reset(self):
        """Clear the internal buffer and confidence history."""
        self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)
        self.buffer_filled = 0
        self.confidence_history.clear()