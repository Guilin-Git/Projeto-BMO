"""
bmo_voice/phrase_bank.py
-------------------------
Banco de frases pré-prontas do BMO indexadas por chave semântica.

Uso atual: texto → sintetizado em tempo real via TTS+RVC.
Uso futuro: os valores podem ser caminhos para arquivos .wav/.mp3
            pré-gravados, pulando o TTS+RVC por completo.

Para expandir: adicione novas chaves ao dicionário PHRASES abaixo,
ou carregue-as de um arquivo JSON/YAML para frases dinâmicas de API.
"""

from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------
# Banco de frases — chave semântica → texto PT-BR
# ------------------------------------------------------------------
PHRASES: dict[str, str] = {
    # Estados gerais
    "greeting":      "Oi! Beemo está aqui!",
    "farewell":      "Até mais! Beemo vai sentir saudade.",
    "thinking":      "Hmm, deixa Beemo pensar um pouquinho.",
    "loading":       "Um segundo, Beemo está carregando.",
    "not_found":     "Beemo não encontrou nada sobre isso.",
    "error":         "Eita... alguma coisa deu errado.",

    # Interações de API / sistema
    "api_success":   "Pronto! Beemo fez isso direitinho.",
    "api_error":     "Ops! A requisição não funcionou.",
    "api_loading":   "Beemo está buscando os dados agora.",

    # Afirmação / negação
    "yes":           "Sim! Beemo acha que sim.",
    "no":            "Não, Beemo acha que não.",
    "unsure":        "Beemo não tem certeza sobre isso.",

    # Reações emocionais
    "happy":         "Yay! Beemo ficou muito feliz!",
    "surprised":     "Nossa! Beemo não esperava isso.",
    "sad":           "Ah... isso deixou Beemo um pouco triste.",
}


def get(key: str, fallback: str | None = None) -> str | None:
    """
    Retorna o texto da frase pelo chave.
    Se a chave não existir, retorna `fallback` (ou None).
    """
    return PHRASES.get(key, fallback)


def all_keys() -> list[str]:
    """Retorna todas as chaves disponíveis no banco."""
    return list(PHRASES.keys())


def add(key: str, text: str) -> None:
    """Adiciona ou sobrescreve uma frase em runtime."""
    PHRASES[key] = text
