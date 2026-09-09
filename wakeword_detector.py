# wakeword_detector.py
import numpy as np
import pyaudio
import threading
import time
from openwakeword.model import Model
from openwakeword.utils import reset

class WakeWordDetector:
    def __init__(self, model_path="models/wakeword/jarvis.tflite", threshold=0.5, chunk_duration=0.08):
        """
        Real-time wake word detector using OpenWakeWord.
        
        Args:
            model_path: Path to your .tflite model file
            threshold: Confidence threshold (0.3-0.7, default 0.5)
            chunk_duration: Audio chunk size in seconds (0.08 = 80ms)
        """
        self.threshold = threshold
        self.chunk_duration = chunk_duration
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * chunk_duration)
        self.is_running = False
        self.callback = None
        self.model = None
        self.audio_stream = None
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the wake word model"""
        try:
            self.model = Model(wakeword_models=[model_path])
            print(f"✅ Wake word model loaded: {model_path}")
            # Get the wake word name from the model
            self.wake_word = list(self.model.prediction_buffer.keys())[0]
            print(f"🔊 Listening for: {self.wake_word}")
        except Exception as e:
            print(f"❌ Failed to load wake word model: {e}")
            self.model = None
    
    def set_callback(self, callback):
        """Set the function to call when wake word is detected"""
        self.callback = callback
    
    def set_threshold(self, threshold):
        """Adjust sensitivity (0.3 = more sensitive, 0.7 = less sensitive)"""
        self.threshold = threshold
        print(f"🔊 Sensitivity set to: {threshold}")
    
    def start(self):
        """Start listening for the wake word in a background thread"""
        if self.model is None:
            print("❌ No model loaded. Cannot start.")
            return False
        
        if self.is_running:
            print("⚠️ Already running.")
            return True
        
        self.is_running = True
        
        # Initialize audio stream
        self.audio_stream = pyaudio.PyAudio()
        self.stream = self.audio_stream.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        print("🎤 Listening for wake word...")
        return True
    
    def stop(self):
        """Stop listening"""
        self.is_running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio_stream') and self.audio_stream:
            self.audio_stream.terminate()
        print("🛑 Wake word detection stopped.")
    
    def _listen(self):
        """Main listening loop (runs in background thread)"""
        while self.is_running:
            try:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Run inference
                prediction = self.model.predict(audio)
                confidence = prediction.get(self.wake_word, 0.0)
                
                # Debug output (optional - comment out if too noisy)
                # print(f"Confidence: {confidence:.2f}", end="\r")
                
                if confidence > self.threshold:
                    print(f"\n🔊 Wake word detected! (confidence: {confidence:.2f})")
                    if self.callback:
                        self.callback(confidence)
                    
                    # Reset prediction buffer to avoid immediate re-triggering
                    reset(self.model)
                    time.sleep(0.5)  # Cooldown to prevent multiple triggers
                
            except Exception as e:
                print(f"⚠️ Error in wake word loop: {e}")
                time.sleep(0.1)
    
    def is_active(self):
        """Check if the detector is running"""
        return self.is_running

# ===== Singleton instance for use in assistant.py =====
_wake_detector = None

def get_wake_detector(model_path="models/wakeword/jarvis.tflite", threshold=0.5):
    """Get or create the singleton wake word detector"""
    global _wake_detector
    if _wake_detector is None:
        _wake_detector = WakeWordDetector(model_path, threshold)
    return _wake_detector

def start_wake_detection(callback, model_path="models/wakeword/jarvis.tflite", threshold=0.5):
    """Convenience function to start the detector with a callback"""
    detector = get_wake_detector(model_path, threshold)
    detector.set_callback(callback)
    return detector.start()

def stop_wake_detection():
    """Stop the detector"""
    global _wake_detector
    if _wake_detector:
        _wake_detector.stop()
        _wake_detector = None