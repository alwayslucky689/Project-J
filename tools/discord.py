# tools/discord.py
"""
Discord-related functions for the AI Assistant
"""

import subprocess
import os

DISCORD_PATH = r"C:\Users\pstef\AppData\Local\Discord\Update.exe"

def open_discord():
    """Opens Discord desktop app"""
    try:
        subprocess.Popen(f'"{DISCORD_PATH}" --processStart Discord.exe', shell=True)
        print("Opened Discord")
        return True
    except Exception as e:
        print(f"❌ Error opening Discord: {e}")
        return False

def check_discord_installed():
    """Check if Discord is installed"""
    return os.path.exists(DISCORD_PATH)

__all__ = [
    'open_discord',
    'check_discord_installed'
]