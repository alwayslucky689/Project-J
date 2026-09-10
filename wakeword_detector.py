# wakeword_detector.py - Fixed with warning suppression and buffer flushing

import numpy as np
import threading
import time as python_time
import sounddevice as sd
import os

# Suppress ONNX Runtime warnings BEFORE importing onnxruntime
import onnxruntime as ort
ort.set_default_logger_severity(3)  # 3 = ERROR only

class WakeWordDetector:
    def __init__(self, model_path="models/wakeword/jarvis_robust_final/jarvis_robust.onnx", 
                 threshold=0.4):
        self.threshold = threshold
        self.sample_rate = 16000
        self.is_running = False
        self.callback = None
        self.session = None
        self.input_name = None
        self.output_name = None
        self.stream = None
        self.cooldown_until = 0
        self.samples_needed = 16000  # 1 second
        self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)
        self.buffer_filled = 0
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the ONNX model"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return
            
            # Session options with error-level logging only
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # Suppress warnings
            
            providers = []
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            
            print(f"✅ ONNX model loaded: {model_path}")
            print(f"   Input shape: {input_shape}")
            print(f"   Output shape: {output_shape}")
            print(f"   Providers: {self.session.get_providers()}")
            
            # [batch_size, 1, 16000] → samples = 16000
            self.samples_needed = input_shape[2] if isinstance(input_shape[2], int) else 16000
            
            self.wake_word = "jarvis"
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            self.session = None
    
    def _predict(self, audio_chunk):
        """Run inference on a 1-second audio chunk"""
        if self.session is None:
            return 0.0
        
        try:
            if len(audio_chunk) < self.samples_needed:
                audio_chunk = np.pad(audio_chunk, (0, self.samples_needed - len(audio_chunk)))
            elif len(audio_chunk) > self.samples_needed:
                audio_chunk = audio_chunk[:self.samples_needed]
            
            # [batch, channels, samples]
            audio_input = audio_chunk.reshape(1, 1, self.samples_needed).astype(np.float32)
            
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: audio_input}
            )
            
            confidence = float(np.array(outputs[0]).flatten()[0])
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            return 0.0
    
    def set_callback(self, callback):
        self.callback = callback
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        print(f"🔊 Sensitivity set to: {threshold}")
    
    def start(self):
        if self.session is None:
            print("❌ No model loaded. Cannot start.")
            return False
        
        if self.is_running:
            return True
        
        self.is_running = True
        self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)
        self.buffer_filled = 0
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=1280,
                callback=self._audio_callback
            )
            self.stream.start()
            print("🎤 Listening for wake word...")
            return True
        except Exception as e:
            print(f"❌ Failed to start: {e}")
            self.is_running = False
            return False
    
    def _audio_callback(self, indata, frames, pa_time, status):
        """Accumulate audio and run inference on full 1-second windows"""
        if not self.is_running:
            return
        
        current_time = python_time.time()
        
        # Always slide the buffer, even during cooldown
        audio = indata.flatten().astype(np.float32) / 32768.0
        
        if len(audio) >= self.samples_needed:
            self.audio_buffer = audio[-self.samples_needed:].copy()
            self.buffer_filled = self.samples_needed
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio))
            self.audio_buffer[-len(audio):] = audio
            self.buffer_filled = min(self.samples_needed, self.buffer_filled + len(audio))
        
        # Skip inference during cooldown
        if current_time < self.cooldown_until:
            return
        
        try:
            if self.buffer_filled >= self.samples_needed:
                confidence = self._predict(self.audio_buffer.copy())
                
                if confidence > self.threshold:
                    print(f"\n🔊 Wake word detected! (confidence: {confidence:.2f})")
                    self.cooldown_until = current_time + 2.0
                    
                    # Flush the buffer so old audio doesn't re-trigger
                    self.audio_buffer = np.zeros(self.samples_needed, dtype=np.float32)
                    self.buffer_filled = 0
                    
                    if self.callback:
                        threading.Thread(
                            target=self.callback, 
                            args=(confidence,), 
                            daemon=True
                        ).start()
                
        except Exception as e:
            if not hasattr(self, '_last_error_time') or python_time.time() - self._last_error_time > 5:
                print(f"⚠️ Error in audio callback: {e}")
                self._last_error_time = python_time.time()
    
    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        print("🛑 Wake word detection stopped.")

# ===== Singleton =====
_wake_detector = None

def start_wake_detection(callback, model_path="models/wakeword/jarvis_robust_final/jarvis_robust.onnx", threshold=0.4):
    global _wake_detector
    if _wake_detector is None:
        _wake_detector = WakeWordDetector(model_path, threshold)
    _wake_detector.set_callback(callback)
    return _wake_detector.start()

def stop_wake_detection():
    global _wake_detector
    if _wake_detector:
        _wake_detector.stop()
        _wake_detector = None