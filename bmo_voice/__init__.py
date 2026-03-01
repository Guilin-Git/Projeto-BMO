"""
bmo_voice/__init__.py
----------------------
Expõe a API pública do módulo bmo_voice.

Uso:
    from bmo_voice import VoiceEngine, VoiceConfig
    import asyncio

    engine = VoiceEngine()
    asyncio.run(engine.speak("Beemo está aqui!"))
"""

from .config import VoiceConfig
from .voice_engine import VoiceEngine
from . import phrase_bank, text_cleaner

__all__ = [
    "VoiceEngine",
    "VoiceConfig",
    "phrase_bank",
    "text_cleaner",
]
