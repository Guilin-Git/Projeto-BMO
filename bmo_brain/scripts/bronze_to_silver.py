"""
bronze_to_silver.py
--------------------
Pipeline Bronze → Silver para o BMO RAG.

Lê os arquivos .md do bronze, estrutura, normaliza e escreve JSON limpo no silver.

Características:
  - IDs estáveis por tipo
  - Normalização de seções via SECTION_MAP
  - Canonicalização de entidades (aliases PT-BR)
  - Extração de listas de personagens de episódios
  - Deduplicação de parágrafos por hash SHA1 (intra-tipo)
  - Schema bem definido por tipo
  - Saída: knowledge/silver/{type}/*.json  +  silver/index.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
BRONZE_DIR  = SCRIPT_DIR.parent / "knowledge" / "bronze"
SILVER_DIR  = SCRIPT_DIR.parent / "knowledge" / "silver"

BRONZE_TYPES = {
    "personagem": BRONZE_DIR / "characters",
    "episodio":   BRONZE_DIR / "episodes",
    "objeto":     BRONZE_DIR / "items",
    "lugar":      BRONZE_DIR / "places",
}

SILVER_TYPES = {
    "personagem": SILVER_DIR / "characters",
    "episodio":   SILVER_DIR / "episodes",
    "objeto":     SILVER_DIR / "items",
    "lugar":      SILVER_DIR / "places",
}

# ---------------------------------------------------------------------------
# Vocabulários de normalização
# ---------------------------------------------------------------------------

# Mapeia variações de nomes de seção → chave canônica (PT-BR)
# Chaves em minúsculas sem acento (aplicadas após slugify leve)
SECTION_MAP: dict[str, str] = {
    # Sinopse / summary
    "sinopse":      "sinopse",
    "resumo":       "sinopse",
    "plot":         "sinopse",

    # Enredo / story
    "enredo":       "enredo",
    "historia":     "historia",
    "story":        "historia",

    # Descrição intro
    "descricao":    "descricao",
    "description":  "descricao",

    # Aparência
    "aparencia":    "aparencia",
    "appearance":   "aparencia",

    # Personalidade
    "personalidade": "personalidade",
    "personality":   "personalidade",

    # Características
    "caracteristicas": "caracteristicas",
    "characteristics": "caracteristicas",

    # Habilidades
    "habilidades":  "habilidades",
    "poderes":      "habilidades",
    "abilities":    "habilidades",
    "powers":       "habilidades",

    # Relacionamentos
    "relacionamentos": "relacionamentos",
    "relationships":   "relacionamentos",

    # Amigos / Inimigos
    "amigos":   "amigos",
    "inimigos": "inimigos",
    "neutro":   "neutro",

    # Personagens (episódios)
    "personagens": "personagens",
    "characters":  "personagens",

    # Encarnações
    "encarnacoes": "encarnacoes",

    # Usuários (objetos)
    "usuarios": "usuarios",

    # Áreas (lugares)
    "areas":         "areas",
    "localizacoes":  "localizacoes",

    # Curiosidades
    "curiosidades": "curiosidades",
    "trivia":       "curiosidades",
    "notas":        "curiosidades",

    # Habitantes / residentes
    "habitantes":   "habitantes",
    "residentes":   "habitantes",
}

# Canonicalização de entidades: alias → nome canônico PT-BR
# Ordem importa: mais específicos primeiro (evitar substituição parcial errada)
ENTITY_CANON: dict[str, str] = {
    # Princesa Jujuba
    "Princess Bubblegum":   "Princesa Jujuba",
    "Bonnibel Bubblegum":   "Princesa Jujuba",
    "Bonnibel":             "Princesa Jujuba",
    "Bonnie":               "Princesa Jujuba",
    r"\bPB\b":              "Princesa Jujuba",

    # Rei Gelado
    "Ice King":  "Rei Gelado",
    "Simon Petrikov": "Simon Petrikov",  # canônico já em PT
    r"\bSimão\b": "Simon Petrikov",

    # Marceline
    "Marceline the Vampire Queen": "Marceline",
    "Vampire Queen":               "Marceline",
    r"\bMarcy\b":                  "Marceline",

    # Lady Íris
    "Lady Íriscornio":        "Lady Íris",
    "Lady Íriscornucópia":    "Lady Íris",
    "Lady Rainicorn":         "Lady Íris",

    # Princesa de Fogo
    "Flame Princess": "Princesa de Fogo",
    "Phoebe":         "Princesa de Fogo",
    r"\bPF\b":        "Princesa de Fogo",

    # BMO
    "Beemo": "BMO",

    # Lugares
    "Candy Kingdom": "Reino Doce",
    r"\bOoo\b":      "Terra de Ooo",

    # Vilões
    "The Lich": "Lich",
    "GOLB":     "GOLB",  # já canônico
}

# ---------------------------------------------------------------------------
# Schemas (dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class SilverBase:
    id:          str
    tipo:        str
    nome:        str
    url:         str
    sections:    dict[str, str]   = field(default_factory=dict)
    source_file: str              = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v or v == 0}


@dataclass
class SilverEpisodio(SilverBase):
    season:               int         = 0
    episode:              int         = 0
    episode_id:           str         = ""   # "S01E001"
    nome_original:        str         = ""
    diretor:              str         = ""
    roteiro:              str         = ""
    storyboard:           str         = ""
    audiencia:            str         = ""
    codigo_producao:      str         = ""
    data_exibicao:        str         = ""
    data_exibicao_br:     str         = ""
    characters_main:      list[str]   = field(default_factory=list)
    characters_secondary: list[str]   = field(default_factory=list)


@dataclass
class SilverPersonagem(SilverBase):
    categoria: str = ""


@dataclass
class SilverLugar(SilverBase):
    tipo_lugar:       str = ""
    localizacao:      str = ""
    governante:       str = ""
    dono:             str = ""
    residentes:       str = ""
    populacao:        str = ""
    status:           str = ""
    primeira_aparicao: str = ""
    ultima_aparicao:  str = ""


@dataclass
class SilverObjeto(SilverBase):
    pass  # extensível — infobox de objetos é muito variado

# ---------------------------------------------------------------------------
# Utilitários gerais
# ---------------------------------------------------------------------------

def _slug_simples(txt: str) -> str:
    """Slug sem acentos, minúsculas, underscores. Usado para SECTION_MAP lookup."""
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    txt = re.sub(r"[^\w\s]", "", txt).strip().lower()
    return re.sub(r"\s+", "_", txt)


def slugify(txt: str) -> str:
    """Slug para nomes de arquivo e IDs."""
    s = _slug_simples(txt)
    return s[:80]


def _sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:12]


def canonicalizar_entidades(txt: str) -> str:
    """Substitui aliases de entidades pelo nome canônico PT-BR."""
    for alias, canonical in ENTITY_CANON.items():
        # Suporte a regex (padrões com \b etc.)
        try:
            txt = re.sub(alias, canonical, txt)
        except re.error:
            txt = txt.replace(alias, canonical)
    return txt


def normalizar_nome_secao(titulo: str) -> str:
    """Traduz título de seção para chave canônica via SECTION_MAP."""
    slug = _slug_simples(titulo)
    return SECTION_MAP.get(slug, slug)

# ---------------------------------------------------------------------------
# Parser de Markdown (nosso formato controlado)
# ---------------------------------------------------------------------------

def parse_md_sections(body: str) -> dict[str, str]:
    """
    Divide o corpo do MD por '## Título' e retorna dict {chave_canônica: conteudo}.
    Preserva h3/h4 dentro do conteúdo de cada seção.
    """
    sections: dict[str, str] = {}

    # Divide por linhas de h2 (## texto)
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    partes   = pattern.split(body)

    # partes[0] = intro antes do primeiro h2 (já extraída como descricao)
    # partes[1,2] = titulo, conteudo
    # partes[3,4] = titulo, conteudo ...
    intro = partes[0].strip()
    if intro:
        sections["descricao"] = intro

    for i in range(1, len(partes) - 1, 2):
        titulo   = partes[i].strip()
        conteudo = partes[i + 1].strip() if i + 1 < len(partes) else ""

        if not titulo:
            continue

        chave = normalizar_nome_secao(titulo)
        if chave in sections:
            # Acumula se já existir (ex: duas seções "Curiosidades")
            sections[chave] += "\n\n" + conteudo
        else:
            sections[chave] = conteudo

    return sections


def extrair_lista_personagens(secao_personagens: str) -> tuple[list[str], list[str]]:
    """
    Dado o conteúdo da seção 'Personagens', extrai listas de principais e secundários.
    Retorna (principais, secundários).
    """
    principais: list[str] = []
    secundarios: list[str] = []
    modo = None

    for linha in secao_personagens.splitlines():
        linha = linha.strip()
        if not linha:
            continue

        if re.match(r"^#{1,4}\s*Principais", linha, re.IGNORECASE):
            modo = "main"
            continue
        if re.match(r"^#{1,4}\s*Secund", linha, re.IGNORECASE):
            modo = "secondary"
            continue
        if re.match(r"^#{1,4}", linha):
            modo = None
            continue

        if linha.startswith("- "):
            nomes_raw = linha[2:].strip()
            # Blobs com múltiplos nomes separados por espaço → dividir por maiúsculas
            # Ex: "Povo Doce Moranguinha Chico Banana" → heurística simples
            nomes = _dividir_blob_nomes(nomes_raw)
            target = principais if modo == "main" else secundarios
            for n in nomes:
                n = canonicalizar_entidades(n.strip())
                if n and n not in target:
                    target.append(n)

    return principais, secundarios


def _dividir_blob_nomes(blob: str) -> list[str]:
    """
    Heurística: se a string for muito longa E não contiver vírgulas,
    pode ser um blob de nomes grudados. Divide por grupos de palavras iniciadas em maiúscula.
    """
    blob = blob.strip()
    if len(blob) <= 40 or "," in blob:
        return [blob]

    # Divide por sequência que começa em maiúscula após espaço
    partes = re.split(r"(?<=[a-záéíóúâêôãõç]) (?=[A-ZÁÉÍÓÚÂÊÔÃÕ])", blob)
    return [p.strip() for p in partes if p.strip()]


def limpar_conteudo(txt: str) -> str:
    """Remove artefatos de texto: citações wiki, âncoras, espaços duplos."""
    txt = re.sub(r"\[\d+\]", "", txt)
    txt = re.sub(r"\[nota \d+\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\[carece de fontes\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\[\s*(editar|edit)\s*\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"> .+\n", "", txt)            # remove linha âncora RAG do bronze
    txt = re.sub(r" {2,}", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# ---------------------------------------------------------------------------
# Deduplicação de parágrafos (intra-tipo)
# ---------------------------------------------------------------------------

class ParaDeduplicator:
    """
    Rastreia hashes de parágrafos. Parágrafos vistos > MAX_REPEATS são
    marcados como lore comum — mantidos mas sinalizados no campo 'type'.
    """
    MAX_REPEATS = 2

    def __init__(self):
        self._counts: dict[str, int] = {}

    def filtrar(self, texto: str) -> str:
        """Remove parágrafos duplicados de um bloco de texto."""
        paragrafos = re.split(r"\n\n+", texto)
        result = []
        for p in paragrafos:
            p = p.strip()
            if not p or len(p) < 40:
                result.append(p)
                continue
            h = _sha1(re.sub(r"\s+", " ", p.lower()))
            self._counts[h] = self._counts.get(h, 0) + 1
            if self._counts[h] <= self.MAX_REPEATS:
                result.append(p)
            # Parágrafos excessivamente repetidos são silenciosamente descartados
        return "\n\n".join(result)

    def processar_sections(self, sections: dict[str, str]) -> dict[str, str]:
        return {k: self.filtrar(v) for k, v in sections.items()}

# ---------------------------------------------------------------------------
# Geradores de ID estável
# ---------------------------------------------------------------------------

def id_episodio(season: int, episode: int) -> str:
    return f"AT_S{season:02d}E{episode:03d}"

def id_personagem(nome: str) -> str:
    return f"AT_CHAR_{slugify(nome)}"

def id_lugar(nome: str) -> str:
    return f"AT_PLACE_{slugify(nome)}"

def id_objeto(nome: str) -> str:
    return f"AT_ITEM_{slugify(nome)}"

# ---------------------------------------------------------------------------
# Processadores por tipo
# ---------------------------------------------------------------------------

def process_episodio(post: frontmatter.Post, source_file: str) -> SilverEpisodio:
    meta     = post.metadata
    nome     = meta.get("nome", "")
    season   = int(meta.get("temporada", 0))
    episode  = int(meta.get("numero_ep", 0))
    sections = parse_md_sections(post.content)

    # Extrai personagens da seção personagens
    chars_main, chars_sec = [], []
    if "personagens" in sections:
        chars_main, chars_sec = extrair_lista_personagens(sections.pop("personagens"))

    # Remove seção descricao redundante (só repete "é o Xº episódio")
    sections.pop("descricao", None)

    # Aplica canonicalização em textos narrativos
    sections = {k: canonicalizar_entidades(limpar_conteudo(v)) for k, v in sections.items()}

    ep = SilverEpisodio(
        id            = id_episodio(season, episode),
        tipo          = "episodio",
        nome          = nome,
        url           = meta.get("url", ""),
        season        = season,
        episode       = episode,
        episode_id    = f"S{season:02d}E{episode:03d}",
        nome_original = meta.get("nome_original", ""),
        diretor       = meta.get("diretor", ""),
        roteiro       = meta.get("roteiro", ""),
        storyboard    = meta.get("storyboard", ""),
        audiencia     = meta.get("audiencia", ""),
        codigo_producao = meta.get("codigo_producao", ""),
        data_exibicao    = meta.get("data_exibicao", ""),
        data_exibicao_br = meta.get("data_exibicao_br", ""),
        characters_main      = chars_main,
        characters_secondary = chars_sec,
        sections    = sections,
        source_file = source_file,
    )
    return ep


def process_personagem(post: frontmatter.Post, source_file: str) -> SilverPersonagem:
    meta     = post.metadata
    nome     = meta.get("nome", "")
    sections = parse_md_sections(post.content)
    sections = {k: canonicalizar_entidades(limpar_conteudo(v)) for k, v in sections.items()}

    return SilverPersonagem(
        id          = id_personagem(nome),
        tipo        = "personagem",
        nome        = nome,
        url         = meta.get("url", ""),
        categoria   = meta.get("categoria", ""),
        sections    = sections,
        source_file = source_file,
    )


def process_lugar(post: frontmatter.Post, source_file: str) -> SilverLugar:
    meta     = post.metadata
    nome     = meta.get("nome", "")
    sections = parse_md_sections(post.content)
    sections = {k: canonicalizar_entidades(limpar_conteudo(v)) for k, v in sections.items()}

    return SilverLugar(
        id                 = id_lugar(nome),
        tipo               = "lugar",
        nome               = nome,
        url                = meta.get("url", ""),
        tipo_lugar         = meta.get("tipo_lugar", ""),
        localizacao        = meta.get("localizacao", ""),
        governante         = meta.get("governante", ""),
        dono               = meta.get("dono", ""),
        residentes         = meta.get("residentes", ""),
        populacao          = meta.get("populacao", ""),
        status             = meta.get("status", ""),
        primeira_aparicao  = meta.get("primeira_aparicao", ""),
        ultima_aparicao    = meta.get("ultima_aparicao", ""),
        sections           = sections,
        source_file        = source_file,
    )


def process_objeto(post: frontmatter.Post, source_file: str) -> SilverObjeto:
    meta     = post.metadata
    nome     = meta.get("nome", "")
    sections = parse_md_sections(post.content)
    sections = {k: canonicalizar_entidades(limpar_conteudo(v)) for k, v in sections.items()}

    return SilverObjeto(
        id          = id_objeto(nome),
        tipo        = "objeto",
        nome        = nome,
        url         = meta.get("url", ""),
        sections    = sections,
        source_file = source_file,
    )


PROCESSORS = {
    "personagem": process_personagem,
    "episodio":   process_episodio,
    "objeto":     process_objeto,
    "lugar":      process_lugar,
}

# ---------------------------------------------------------------------------
# Runner por tipo
# ---------------------------------------------------------------------------

def _load_post_safe(filepath: Path) -> frontmatter.Post:
    """
    Tenta carregar com python-frontmatter. Se falhar (ex: YAML inválido gerado
    pelo scraper com aspas soltas), faz um fallback parsing usando regex simples
    para garantir que 100% dos arquivos passem para a camada silver.
    """
    try:
        return frontmatter.load(str(filepath))
    except Exception:
        # Fallback resiliente: lê linha por linha e extrai chave: valor na marra
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            
        match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if not match:
            return frontmatter.Post(content) # Sem frontmatter reconhecível
            
        yaml_text = match.group(1)
        body_text = match.group(2)
        
        metadata = {}
        for line in yaml_text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                metadata[key] = val
                
        return frontmatter.Post(body_text, **metadata)

def processar_tipo(tipo: str, dedup: ParaDeduplicator) -> list[dict]:
    bronze_path = BRONZE_TYPES[tipo]
    silver_path = SILVER_TYPES[tipo]
    silver_path.mkdir(parents=True, exist_ok=True)

    processor = PROCESSORS[tipo]
    arquivos  = sorted(bronze_path.glob("*.md"))
    resultados = []
    erros      = []

    print(f"\n  📂  {tipo.upper()} — {len(arquivos)} arquivos")
    for f in arquivos:
        try:
            post   = _load_post_safe(f)
            record = processor(post, f.name)

            # Deduplicação de parágrafos (altera sections in-place)
            record.sections = dedup.processar_sections(record.sections)

            d = record.to_dict()
            resultados.append(d)

            # Salva JSON individual
            out_path = silver_path / f"{record.id}.json"
            with open(out_path, "w", encoding="utf-8") as fout:
                json.dump(d, fout, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"    ✗ {f.name}: {e}")
            erros.append(f.name)

    ok = len(resultados) - len(erros)
    print(f"    ✔ {ok} processados  |  ✗ {len(erros)} erros")
    return resultados

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("  Bronze → Silver Pipeline  |  BMO RAG")
    print(f"{'='*60}")

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    todos: dict[str, list[dict]] = {}
    index: list[dict]            = []

    for tipo in ["personagem", "episodio", "objeto", "lugar"]:
        dedup      = ParaDeduplicator()  # dedup independente por tipo
        registros  = processar_tipo(tipo, dedup)
        todos[tipo] = registros

        # Adiciona ao índice global (campos mínimos)
        folder_name = SILVER_TYPES[tipo].name
        for r in registros:
            index.append({
                "id":   r["id"],
                "tipo": r["tipo"],
                "nome": r["nome"],
                "file": f"{folder_name}/{r['id']}.json",
            })

    # Índice global
    index_path = SILVER_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # Sumário por tipo
    totais_silver = {t: len(v) for t, v in todos.items()}

    print(f"\n{'='*60}")
    print("  Resumo Silver")
    print(f"{'='*60}")
    for tipo, n in totais_silver.items():
        print(f"  {tipo:<15} → {n:>4} registros")
    print(f"  {'TOTAL':<15} → {sum(totais_silver.values()):>4} registros")
    print(f"  Índice global : {index_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
