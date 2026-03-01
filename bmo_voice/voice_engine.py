"""
bmo_voice/voice_engine.py
--------------------------
Núcleo do sistema de voz do BMO.

Pipeline de baixa latência (por sentença):
    texto → TextCleaner → SentenceSplitter
         → Queue → TTSWorker (edge-tts)
         → Queue → RVCWorker (Applio via gradio-client)
         → Queue → PlaybackWorker (sounddevice)

Cada estágio roda de forma concorrente via asyncio.
O áudio começa a tocar enquanto frases posteriores são processadas.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from . import text_cleaner, phrase_bank
from .config import VoiceConfig
from .tts_backends.edge_tts_backend import EdgeTTSBackend
from .rvc_backends.applio_backend import ApplioBackend

log = logging.getLogger(__name__)

# Sentinela de fim de pipeline
_DONE = object()


class VoiceEngine:
    """
    Entrypoint público do sistema de voz do BMO.

    Uso:
        engine = VoiceEngine()
        asyncio.run(engine.speak("Beemo está aqui!"))
        asyncio.run(engine.speak_phrase("greeting"))
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        cfg = self.config

        self._tts = EdgeTTSBackend(
            voice=cfg.tts_voice,
            rate=cfg.tts_rate,
            volume=cfg.tts_volume,
            pitch=cfg.tts_pitch,
        )

        self._rvc = ApplioBackend(
            applio_url=cfg.applio_url,
            pth_path=cfg.rvc_model,
            index_path=cfg.rvc_index,
            pitch=cfg.rvc_pitch,
            index_rate=cfg.rvc_index_rate,
            volume_envelope=cfg.rvc_volume_envelope,
            protect=cfg.rvc_protect,
            f0_method=cfg.rvc_f0_method,
            clean_audio=cfg.rvc_clean_audio,
            clean_strength=cfg.rvc_clean_strength,
            split_audio=cfg.rvc_split_audio,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Sintetiza e fala o texto via pipeline TTS → RVC."""
        sentences = text_cleaner.split_sentences(text)
        if not sentences:
            log.warning("speak() chamado com texto vazio.")
            return
        await self._run_pipeline(sentences)

    async def speak_phrase(self, key: str) -> None:
        """
        Fala uma frase pré-pronta do PhraseBank por chave semântica.
        Futuro: se o valor for caminho de .wav/.mp3, toca direto.
        """
        text = phrase_bank.get(key)
        if text is None:
            log.warning("Chave '%s' não encontrada no PhraseBank.", key)
            return
        await self.speak(text)

    async def close(self) -> None:
        """Libera recursos."""
        self._rvc.close()

    # ------------------------------------------------------------------
    # Pipeline interno
    # ------------------------------------------------------------------

    async def _run_pipeline(self, sentences: list[str]) -> None:
        """
        Executa as 3 etapas em paralelo via asyncio.Queue.
        Sentença N+1 entra no TTS enquanto sentença N está no RVC.
        """
        cfg = self.config
        q_tts: asyncio.Queue = asyncio.Queue(maxsize=cfg.queue_maxsize)
        q_rvc: asyncio.Queue = asyncio.Queue(maxsize=cfg.queue_maxsize)

        await asyncio.gather(
            self._tts_worker(sentences, q_tts),
            self._rvc_worker(q_tts, q_rvc),
            self._playback_worker(q_rvc),
        )

    async def _tts_worker(
        self, sentences: list[str], out_q: asyncio.Queue
    ) -> None:
        """Converte cada sentença em WAV usando edge-tts."""
        try:
            for sentence in sentences:
                if not sentence.strip():
                    continue
                log.debug("TTS: '%s'", sentence[:60])
                wav_bytes, sr = await self._tts.synthesize(sentence)
                await out_q.put((wav_bytes, sr))
        finally:
            await out_q.put(_DONE)

    async def _rvc_worker(
        self, in_q: asyncio.Queue, out_q: asyncio.Queue
    ) -> None:
        """Aplica RVC em cada chunk de áudio via Applio."""
        try:
            while True:
                item = await in_q.get()
                if item is _DONE:
                    break
                wav_bytes, sr = item
                log.debug("RVC: chunk de %d bytes → Applio", len(wav_bytes))
                audio_np, out_sr = await self._rvc.convert(wav_bytes, sr)
                await out_q.put((audio_np, out_sr))
        finally:
            await out_q.put(_DONE)

    async def _playback_worker(self, in_q: asyncio.Queue) -> None:
        """Toca os chunks de áudio convertidos em sequência."""
        loop = asyncio.get_event_loop()
        while True:
            item = await in_q.get()
            if item is _DONE:
                break
            audio_np, sr = item
            log.debug("Playback: %d samples @ %d Hz", len(audio_np), sr)
            await loop.run_in_executor(None, self._play_sync, audio_np, sr)

    def _play_sync(self, audio: np.ndarray, sr: int) -> None:
        """Toca numpy array de áudio via sounddevice (síncrono)."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        sd.play(audio, samplerate=sr)
        sd.wait()
