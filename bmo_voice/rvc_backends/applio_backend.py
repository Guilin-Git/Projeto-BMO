"""
bmo_voice/rvc_backends/applio_backend.py
------------------------------------------
Backend RVC usando o Applio via gradio-client.

Pré-requisito:
    1. Applio instalado e rodando localmente:
       python app.py --port 7865
    2. gradio-client instalado:
       uv add gradio-client

O Applio expõe sua API de inferência via Gradio.
Usamos gradio_client para chamar o endpoint /infer_convert
com os arquivos de modelo BMO já presentes em bmo_voice/voice_models/.

Latência estimada: 300ms–1s por sentença (depende do hardware).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# Thread pool dedicado ao Applio (I/O + computação)
_applio_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="applio")


class ApplioBackend:
    """
    Aplica conversão de voz RVC via Applio rodando localmente.

    Interface:
        audio_np, sr = await backend.convert(wav_bytes, input_sr)
    """

    def __init__(
        self,
        applio_url: str,
        pth_path: str,
        index_path: str,
        pitch: int = 0,
        index_rate: float = 0.75,
        volume_envelope: float = 0.25,
        protect: float = 0.33,
        f0_method: str = "rmvpe",
        clean_audio: bool = True,
        clean_strength: float = 0.5,
        split_audio: bool = False,
        export_format: str = "WAV",
    ) -> None:
        self.applio_url = applio_url
        self.pth_path = pth_path
        self.index_path = index_path
        self.pitch = pitch
        self.index_rate = index_rate
        self.volume_envelope = volume_envelope
        self.protect = protect
        self.f0_method = f0_method
        self.clean_audio = clean_audio
        self.clean_strength = clean_strength
        self.split_audio = split_audio
        self.export_format = export_format
        self._client = None  # gradio_client — inicialização lazy

    def _get_client(self):
        """Inicializa o gradio_client lazy na primeira chamada."""
        if self._client is None:
            from gradio_client import Client
            log.info("Conectando ao Applio em %s ...", self.applio_url)
            self._client = Client(self.applio_url)
            log.info("Conectado ao Applio.")
        return self._client

    async def convert(self, wav_bytes: bytes, input_sr: int) -> tuple[np.ndarray, int]:
        """
        Converte áudio WAV usando o RVC do Applio.
        Roda na thread pool para não bloquear o asyncio.

        Returns:
            (audio_numpy_float32, sample_rate)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _applio_executor,
            self._convert_sync,
            wav_bytes,
            input_sr,
        )

    def _convert_sync(self, wav_bytes: bytes, sr: int) -> tuple[np.ndarray, int]:
        # Identifica a pasta do Applio
        applio_dir = r"C:\Users\PC\ApplioV3.6.2"
        python_exe = os.path.join(applio_dir, "env", "python.exe")

        if not os.path.exists(python_exe):
            raise RuntimeError(f"Python do Applio não encontrado em {python_exe}. Garanta que o caminho está correto.")

        # Cria paths temporários
        fd_in, tmp_in_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_in)

        fd_out, tmp_out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_out)

        try:
            with open(tmp_in_path, "wb") as f:
                f.write(wav_bytes)

            import subprocess
            
            # Script de inferência usando o core do Applio
            script = f"""
import sys
sys.path.append(r"{applio_dir}")
from core import run_infer_script

# run_infer_script signature
# pitch, filter_radius, index_rate, volume_envelope, protect,
# f0_method, audio_path, output_path, model_file, index_file ...
res = run_infer_script(
    {self.pitch}, # pitch: int
    {self.index_rate}, # index_rate: float
    {self.volume_envelope}, # volume_envelope: float
    {self.protect}, # protect: float
    '{self.f0_method}', # f0_method: str
    r'{tmp_in_path}', # input_path: str
    r'{tmp_out_path}', # output_path: str
    r'{self.pth_path}', # pth_path: str
    r'{self.index_path}', # index_path: str
    False, # split_audio: bool
    False, # f0_autotune: bool
    1, # f0_autotune_strength: float
    False, # proposed_pitch: bool
    155.0, # proposed_pitch_threshold: float
    {self.clean_audio}, # clean_audio: bool
    {self.clean_strength}, # clean_strength: float
    '{self.export_format}', # export_format: str
    "contentvec", # embedder_model: str
    "", # embedder_model_custom: str
    False, # formant_shifting: bool
    1.0, # formant_qfrency: float
    1.0, # formant_timbre: float
    False, # post_process: bool
    False, # reverb: bool
    False, # pitch_shift: bool
    False, # limiter: bool
    False, # gain: bool
    False, # distortion: bool
    False, # chorus: bool
    False, # bitcrush: bool
    False, # clipping: bool
    False, # compressor: bool
    False, # delay: bool
    0.5, # reverb_room_size: float
    0.5, # reverb_damping: float
    0.5, # reverb_wet_gain: float
    0.5, # reverb_dry_gain: float
    0.5, # reverb_width: float
    0.5, # reverb_freeze_mode: float
    0.0, # pitch_shift_semitones: float
    -6, # limiter_threshold: float
    0.01, # limiter_release_time: float
    0.0, # gain_db: float
    25, # distortion_gain: float
    1.0, # chorus_rate: float
    0.25, # chorus_depth: float
    7, # chorus_center_delay: float
    0.0, # chorus_feedback: float
    0.5, # chorus_mix: float
    8, # bitcrush_bit_depth: int
    -6, # clipping_threshold: float
    0, # compressor_threshold: float
    1, # compressor_ratio: float
    1.0, # compressor_attack: float
    100, # compressor_release: float
    0.5, # delay_seconds: float
    0.0, # delay_feedback: float
    0.5, # delay_mix: float
    0 # sid: int
)
print("APP_RES:", res)
"""
            script_path = tmp_in_path.replace(".wav", "_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

            log.debug(f"Applio CLI: chamando inferência com script temporário em {applio_dir}")

            # Roda no env do Applio
            result = subprocess.run(
                [python_exe, script_path],
                cwd=applio_dir,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                log.error(f"Erro Applio stdout: {result.stdout}")
                log.error(f"Erro Applio stderr: {result.stderr}")
                raise RuntimeError(f"Applio CLI falhou. Erro: {result.stderr}")

            if not os.path.exists(tmp_out_path):
                raise RuntimeError(f"Applio não gerou arquivo de saída: {result.stdout}")
            
            if os.path.getsize(tmp_out_path) == 0:
                 raise RuntimeError(f"Applio gerou arquivo de áudio vazio. Log: {result.stdout}")

            audio, sr_out = sf.read(tmp_out_path, dtype="float32")
            
            # Limpa script
            Path(script_path).unlink(missing_ok=True)
            
            return audio, sr_out

        finally:
            Path(tmp_in_path).unlink(missing_ok=True)
            # Não remove o output — Applio pode gerenciar esse path

    def close(self) -> None:
        _applio_executor.shutdown(wait=False)
