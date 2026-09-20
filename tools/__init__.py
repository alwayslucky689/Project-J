# tools/__init__.py
"""
Tools package for the AI Assistant.

Importing this package triggers registration of every tool into
core.registry.TOOL_REGISTRY. Do not remove the submodule imports below,
even if they appear unused — they exist for their side effects.
"""
from tools import youtube       # noqa: F401
from tools import spotify       # noqa: F401
from tools import discord       # noqa: F401
from tools import ookla         # noqa: F401
from tools import ollama        # noqa: F401
from tools import personality_tools  # noqa: F401
from tools import tts_tools     # noqa: F401

#def __init__(self, ..., threshold=0.35, ...):
#self.volume = 0.8  # Default 80%
#self.volume = 0.8  # Default volume (0.0 to 1.0)
#pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
#self.confidence_history = deque(maxlen=3)