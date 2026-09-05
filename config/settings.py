# config/settings.py
"""
Configuration settings for the AI Assistant
"""

import os

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Paths
DISCORD_PATH = r"C:\Users\pstef\AppData\Local\Discord\Update.exe"

# YouTube settings
YOUTUBE_MAX_RESULTS = 5

# Spotify API Credentials - ADD YOURS HERE
SPOTIFY_CLIENT_ID = "951c6db56b774bd58d35cc019779d0f3"  # Replace with your Client ID
SPOTIFY_CLIENT_SECRET = "840b8913eb88425bb7e9553c7c86286d"  # Replace with your Client Secret
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"

# Debug mode
DEBUG = True

# Exit commands
EXIT_COMMANDS = ["quit", "bye", "exit", "cao", "kys"]

__all__ = [
    'OLLAMA_URL',
    'OLLAMA_MODEL',
    'DISCORD_PATH',
    'YOUTUBE_MAX_RESULTS',
    'DEBUG',
    'EXIT_COMMANDS',
    'SPOTIFY_CLIENT_ID',
    'SPOTIFY_CLIENT_SECRET',
    'SPOTIFY_REDIRECT_URI'
]