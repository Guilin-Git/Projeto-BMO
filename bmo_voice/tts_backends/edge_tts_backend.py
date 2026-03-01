"""
bmo_voice/tts_backends/edge_tts_backend.py
--------------------------------------------
Backend TTS usando edge-tts (Microsoft Azure Neural TTS via rede).
100% compatível com Python 3.13, sem dependências nativas.

Voz PT-BR recomendadas:
  - pt-BR-ThalitaNeural  (feminina, suave)
  - pt-BR-AntonioNeural  (masculino, neutro)
  - pt-BR-FranciscaNeural (feminina, expressiva)
"""

from __future__ import annotations

import asyncio
import io
import wave

import edge_tts
import numpy as np
import soundfile as sf

from .base_backend import TTSBackend


class EdgeTTSBackend(TTSBackend):
    """
    Backend TTS usando edge-tts.
    Gera áudio MP3 via stream e converte para WAV PCM em memória,
    pronto para o RVC processar.
    """

    def __init__(
        self,
        voice: str = "pt-BR-ThalitaNeural",
        rate: str = "+5%",
        volume: str = "+0%",
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.volume = volume

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """
        Sintetiza texto em áudio WAV PCM usando edge-tts.

        Returns:
            (wav_bytes, sample_rate)
        """
        # Gera MP3 em memória via streaming
        mp3_buf = io.BytesIO()
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, volume=self.volume
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])

        mp3_buf.seek(0)
        mp3_bytes = mp3_buf.getvalue()

        if not mp3_bytes:
            raise RuntimeError("edge-tts não retornou áudio. Verifique conexão de rede.")

        # Converte MP3 → numpy float32 usando soundfile
        audio_np, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")

        # Converte float32 array → WAV PCM bytes (16-bit)
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_np, sr, format="WAV", subtype="PCM_16")
        wav_buf.seek(0)

        return wav_buf.read(), sr
