"""
bmo_voice/config.py
-------------------
Configuração central do sistema de voz do BMO.
Altere aqui para controlar voz, velocidade, Applio URL e parâmetros RVC.
"""

from dataclasses import dataclass
from pathlib import Path

_BASE = Path(__file__).parent


@dataclass
class VoiceConfig:
    # ------------------------------------------------------------------
    # TTS — edge-tts (Microsoft Neural TTS via rede, sem instalação nativa)
    # ------------------------------------------------------------------
    # Vozes PT-BR disponíveis:
    #   pt-BR-ThalitaNeural   (feminina, suave)       ← padrão
    #   pt-BR-AntonioNeural   (masculino, neutro)
    #   pt-BR-FranciscaNeural (feminina, expressiva)
    tts_voice: str = "pt-BR-ThalitaNeural"
    tts_rate: str = "+5%"     # velocidade: +X% = mais rápido, -X% = mais lento
    tts_volume: str = "+0%"

    # ------------------------------------------------------------------
    # RVC — Applio rodando localmente via Gradio API
    # ------------------------------------------------------------------
    # URL do Applio (padrão: porta 7865)
    applio_url: str = "http://localhost:7865"

    # Caminhos dos modelos BMO (relativos à raiz do projeto)
    rvc_model: str = str(_BASE / "voice_models" / "BMOcvlc1.5V1_135e_2430s.pth")
    rvc_index: str = str(_BASE / "voice_models" / "BMOcvlc1.5V1.index")

    # Pitch shift em semitons (0 = sem alteração de tom)
    rvc_pitch: int = 0

    # Quanto do índice de features usar (0.0–1.0)
    rvc_index_rate: float = 0.75

    # Método de extração de f0
    # "rmvpe" = melhor qualidade | "harvest" = mais rápido | "crepe" = alternativa
    rvc_f0_method: str = "rmvpe"

    # Proteção de vogais/consoantes (0.0–0.5)
    rvc_protect: float = 0.33

    # Volume envelope mix (0.0–1.0)
    rvc_volume_envelope: float = 0.25

    # Post-process: limpeza de artefatos de áudio
    rvc_clean_audio: bool = True
    rvc_clean_strength: float = 0.5

    # Dividir áudio longo em partes (reduz VRAM, aumenta latência)
    rvc_split_audio: bool = False

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    queue_maxsize: int = 8
