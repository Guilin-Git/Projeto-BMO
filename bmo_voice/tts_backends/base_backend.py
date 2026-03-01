"""
bmo_voice/tts_backends/base_backend.py
----------------------------------------
Interface abstrata para backends de TTS.
Implemente esta classe para adicionar um novo motor de síntese
sem alterar nenhuma outra parte do sistema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSBackend(ABC):
    """
    Contrato que todo backend de TTS deve implementar.

    synthesize() recebe uma string e retorna uma tupla:
        (audio_bytes: bytes, sample_rate: int)

    Os bytes devem estar em formato WAV PCM (16-bit, mono ou stereo)
    para compatibilidade direta com o RVCWorker.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """
        Sintetiza `text` em áudio.

        Returns:
            (wav_bytes, sample_rate)
        """
        ...

    async def close(self) -> None:
        """
        Libera recursos (opcional – sobrescreva se necessário).
        Chamado quando o VoiceEngine é encerrado.
        """
