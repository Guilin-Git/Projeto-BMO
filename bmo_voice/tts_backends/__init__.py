"""
bmo_voice/tts_backends/__init__.py
"""
from .base_backend import TTSBackend
from .edge_tts_backend import EdgeTTSBackend

__all__ = ["TTSBackend", "EdgeTTSBackend"]

