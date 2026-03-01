"""
bmo_voice/text_cleaner.py
--------------------------
Remove artefatos de texto que soariam mal no TTS:
markdown, emojis, símbolos especiais, etc.
Também faz o split em sentenças para alimentar o pipeline.
"""

import re
import unicodedata

# ------------------------------------------------------------------
# Padrões de limpeza
# ------------------------------------------------------------------

# Remove blocos de código e inline code
_RE_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`[^`]+`")

# Remove markdown: **, *, __, _, ~~ etc.
_RE_MARKDOWN = re.compile(r"(\*{1,3}|_{1,3}|~~)")

# Remove links markdown [texto](url) → mantém "texto"
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")

# Remove URLs brutas
_RE_URL = re.compile(r"https?://\S+")

# Remove múltiplos espaços / linhas em branco
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NL = re.compile(r"\n{2,}")

# Quebra de sentença: ponto, exclamação, interrogação seguidos de espaço/fim
_RE_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÀÃÕÂÊÎÔÛÇ\"])|(?<=[.!?…])$",
    re.MULTILINE,
)

# Abreviações comuns PT-BR que NÃO devem ser quebradas em sentença
_ABBREVIATIONS = {
    "sr.", "sra.", "dr.", "dra.", "prof.", "profa.",
    "ex.", "etc.", "obs.", "ref.", "pág.", "vol.",
    "av.", "al.", "r.", "km.", "cm.", "kg.", "g.",
}


def _is_emoji(char: str) -> bool:
    """Retorna True se o caractere é um emoji ou símbolo especial."""
    cat = unicodedata.category(char)
    cp = ord(char)
    # Faixas de emoji Unicode
    return (
        cat in ("So", "Sm", "Sk")
        or 0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0xFE00 <= cp <= 0xFE0F
    )


def clean(text: str) -> str:
    """
    Limpa o texto para TTS: remove markdown, emojis e normaliza espaços.
    Retorna string limpa.
    """
    # Remove blocos de código
    text = _RE_CODE_BLOCK.sub(" ", text)
    text = _RE_INLINE_CODE.sub(" ", text)

    # Extrai texto de links markdown
    text = _RE_LINK.sub(r"\1", text)

    # Remove URLs
    text = _RE_URL.sub("", text)

    # Remove marcações markdown
    text = _RE_MARKDOWN.sub("", text)

    # Remove emojis caractere a caractere
    text = "".join(c for c in text if not _is_emoji(c))

    # Normaliza quebras de linha → espaço
    text = text.replace("\n", " ").replace("\r", " ")

    # Normaliza espaços múltiplos
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Divide o texto em sentenças para alimentar o pipeline frase a frase.
    Garante que cada sentença tem conteúdo mínimo para o TTS processar.
    """
    text = clean(text)
    if not text:
        return []

    # Split simples em pontuação terminal + espaço
    raw = re.split(r"(?<=[.!?…])\s+", text)

    sentences: list[str] = []
    buffer = ""

    for part in raw:
        part = part.strip()
        if not part:
            continue

        # Verifica se termina com abreviação conhecida (não é fim de sentença real)
        lower = part.lower()
        is_abbrev = any(lower.endswith(abbr) for abbr in _ABBREVIATIONS)

        if is_abbrev:
            buffer = (buffer + " " + part).strip()
        else:
            combined = (buffer + " " + part).strip()
            if len(combined) >= 3:  # mínimo de 3 chars para valer a pena sintetizar
                sentences.append(combined)
            buffer = ""

    # Flush do buffer restante
    if buffer.strip():
        sentences.append(buffer.strip())

    return sentences
