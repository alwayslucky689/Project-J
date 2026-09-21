"""
TTS control tools — registered as AI-callable tools.

Wraps the module-level functions in tts.tts_manager so they can be
dispatched through the registry like any other tool.
"""
from core.registry import register, Tool, Permission
from core.tool_descriptions import DESCRIPTIONS
from tts.tts_manager import (
    mute_tts,
    unmute_tts,
    toggle_mute,
    is_muted,
    set_tts_volume,
    get_tts_volume,
)


# ===== Handlers =====

def _handle_mute():
    mute_tts()
    return "🔇 Voice output muted. I'll only respond in text."


def _handle_unmute():
    unmute_tts()
    return "🔊 Voice output enabled."


def _handle_toggle():
    toggle_mute()
    current = "muted" if is_muted() else "enabled"
    return f"🔊 Voice output {current}."


def _handle_set_volume(volume=70):
    set_tts_volume(volume)
    return f"🔊 TTS volume set to {volume}%"


def _handle_raise_volume(amount=10):
    current = get_tts_volume()
    new_vol = min(100, current + amount)
    set_tts_volume(new_vol)
    return f"🔊 TTS volume increased to {new_vol}%"


def _handle_lower_volume(amount=10):
    current = get_tts_volume()
    new_vol = max(0, current - amount)
    set_tts_volume(new_vol)
    return f"🔊 TTS volume decreased to {new_vol}%"


# ===== Registration =====

register(Tool(
    name="mute_tts",
    description=DESCRIPTIONS["mute_tts"],
    handler=_handle_mute,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="tts_muted",
))

register(Tool(
    name="unmute_tts",
    description=DESCRIPTIONS["unmute_tts"],
    handler=_handle_unmute,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="tts_unmuted",
))

register(Tool(
    name="toggle_tts",
    description=DESCRIPTIONS["toggle_tts"],
    handler=_handle_toggle,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="tts_toggled",
))

register(Tool(
    name="set_tts_volume",
    description=DESCRIPTIONS["set_tts_volume"],
    handler=_handle_set_volume,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="tts_volume_set",
))

register(Tool(
    name="raise_tts_volume",
    description=DESCRIPTIONS["raise_tts_volume"],
    handler=_handle_raise_volume,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="tts_volume_raised",
))

register(Tool(
    name="lower_tts_volume",
    description=DESCRIPTIONS["lower_tts_volume"],
    handler=_handle_lower_volume,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="tts_volume_lowered",
))