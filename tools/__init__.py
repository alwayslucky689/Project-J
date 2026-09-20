# tools/__init__.py
"""
Tools package for the AI Assistant
"""
from tts import tts_manager
from tools import youtube
from tools import spotify
from collections import deque
from tools import discord
from tools import ollama
from tools import ookla

#def __init__(self, ..., threshold=0.35, ...):
#self.volume = 0.8  # Default 80%
#self.volume = 0.8  # Default volume (0.0 to 1.0)
#pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
#self.confidence_history = deque(maxlen=3)