"""
bmo_voice/tts_backends/piper_backend.py
-----------------------------------------
Backend TTS usando Piper (local, PT-BR, ~80-150ms por sentença).

Prerequisito:
    - pip install piper-tts
    - Baixar modelo:
        python -m piper --download-dir bmo_voice/tts_models \
               --model pt_BR-faber-medium

O arquivo .onnx + .onnx.json devem estar em bmo_voice/tts_models/.
"""

from __future__ import annotations

import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from .base_backend import TTSBackend

if TYPE_CHECKING:
    from piper.voice import PiperVoice

# Thread pool dedicado ao Piper (CPU-bound, não bloqueia o event loop)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper")


class PiperBackend(TTSBackend):
    """Backend de TTS usando Piper — totalmente local e rápido."""

    def __init__(
        self,
        model_path: str | Path,
        speaker_id: int = 0,
        length_scale: float = 0.95,
    ) -> None:
        self._model_path = Path(model_path)
        self._speaker_id = speaker_id
        self._length_scale = length_scale
        self._voice: PiperVoice | None = None

    # ------------------------------------------------------------------
    # Inicialização lazy — carrega o modelo apenas na primeira chamada
    # ------------------------------------------------------------------

    def _load_voice(self) -> "PiperVoice":
        """Carrega o modelo Piper de forma síncrona (chamado na thread pool)."""
        if self._voice is None:
            from piper.voice import PiperVoice  # import tardio para evitar overhead

            self._voice = PiperVoice.load(str(self._model_path))
        return self._voice

    # ------------------------------------------------------------------
    # TTSBackend interface
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """
        Sintetiza `text` com Piper de forma assíncrona.
        A síntese roda na thread pool para não bloquear o asyncio.

        Returns:
            (wav_pcm_bytes, sample_rate)
        """
        loop = asyncio.get_event_loop()
        wav_bytes, sr = await loop.run_in_executor(
            _executor,
            self._synthesize_sync,
            text,
        )
        return wav_bytes, sr

    def _synthesize_sync(self, text: str) -> tuple[bytes, int]:
        """Síntese síncrona — roda na thread pool."""
        voice = self._load_voice()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            voice.synthesize(
                text,
                wav,
                speaker_id=self._speaker_id,
                length_scale=self._length_scale,
            )

        sample_rate = voice.config.sample_rate
        return buf.getvalue(), sample_rate

    async def close(self) -> None:
        """Libera thread pool."""
        _executor.shutdown(wait=False)
