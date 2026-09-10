# tools/__init__.py
"""
Tools package for the AI Assistant
"""
import tts_manager
from tools import history
from tools import youtube
from tools import spotify
from tools import discord
from tools import ollama
from tools import ookla
import pygame
__all__ = [
    'open_spotify',
    'remember',
    'recall',
    'play_spotify_song',
    'queue_spotify_song',
    'play_spotify_playlist',
    'play_my_playlist',  # Add this
    'play_spotify',
    'list_playlists',
    'raise_volume',
    'lower_volume',
    'set_volume',
    'mute_volume',
    'unmute_volume',
    'pause_spotify',
    'resume_spotify',
    'next_track',
    'previous_track',
    'get_current_track',
    'clear_queue'
]
#self.volume = 0.8  # Default 80%
#self.volume = 0.8  # Default volume (0.0 to 1.0)
#pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)