# tools/discord.py
"""
Discord-related functions for the AI Assistant
"""
from core.registry import register, Tool, Permission
import subprocess
import os
from config.paths import DISCORD_EXE
from core.tool_descriptions import DESCRIPTIONS
DISCORD_PATH = DISCORD_EXE

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
def _fmt_open_discord(result):
    return "✅ Opened Discord" if result else "❌ Failed to open Discord"


register(Tool(
    name="open_discord",
    description=DESCRIPTIONS["open_discord"],
    handler=open_discord,
    formatter=_fmt_open_discord,
    permission=Permission.ACTION,
    response_key="app_opened",
))