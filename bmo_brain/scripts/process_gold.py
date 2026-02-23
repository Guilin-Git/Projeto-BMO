"""
process_gold.py
---------------
Camada Gold do pipeline de qualidade de dados do BMO Brain.
Lê os Markdowns limpos de knowledge/silver/ e aplica enriquecimentos
otimizados para RAG, salvando o resultado em knowledge/gold/.

Enriquecimentos aplicados:
  1. Adicionar campo 'resumo' no frontmatter (primeiro parágrafo, ≤200 chars)
  2. Tags semânticas derivadas da categoria/conteúdo
  3. Cross-referencing: episódios mencionados, personagens referenciados
  4. Contexto de chunking para ajudar no retrieval
  5. Filtrar documentos incompletos (mantidos mas sinalizados)
"""

import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

BASE_DIR = r"C:\Users\PC\Projeto-BMO\bmo_brain\knowledge"
SILVER_DIR = os.path.join(BASE_DIR, "silver")
GOLD_DIR = os.path.join(BASE_DIR, "gold")

CATEGORIAS = ["episodes", "characters", "places", "items"]

# ---------------------------------------------------------------------------
# Personagens conhecidos para cross-referencing
# ---------------------------------------------------------------------------

PERSONAGENS_CONHECIDOS = [
    "Finn", "Jake", "BMO", "Marceline", "Princesa Jujuba", "Rei Gelado",
    "Simon Petrikov", "Lich", "Hunson Abadeer", "Lady Íris", "Lady Iris",
    "Lady Arco-Íris", "Gunter", "Princesa de Fogo", "Princesa Caroço",
    "Neptr", "Prismo", "Morte", "Billy", "Martin", "Martin Mertens",
    "Fern", "Tronquinho", "Conde de Limonágrab", "Princesa Biscoito",
    "Abracadaniel", "Chicletão", "Shelby", "Peppermint Butler",
    "Mordomo Menta", "Joshua", "Margaret", "Jermaine", "Canyon",
    "Rei da Luta", "Rei de Ooo", "Golb", "Betty", "Betty Grof",
    "Evergreen", "Minerva", "Susana Forte", "Ash", "Braco",
    "Kim Kil Whan", "Jake Jr", "Charlie", "Viola",
    "Dona Tromba", "Caracol", "Shoko", "Hugo", "Frieda",
    "Coruja Cósmica", "Maga Caçadora", "Choose Goose",
]

# Tags derivadas de categorias de personagens
TAGS_CATEGORIA = {
    "Protagonistas": ["protagonista", "herói"],
    "Vilões": ["vilão", "antagonista"],
    "Princesas": ["princesa", "realeza"],
    "Família do Finn": ["família", "finn"],
    "Família do Jake": ["família", "jake"],
    "Entidades Cósmicas": ["entidade_cósmica", "mágico"],
    "Povo Doce": ["povo_doce", "reino_doce"],
    "Amigos/Aliados": ["aliado", "amigo"],
    "Reis e Líderes": ["líder", "realeza"],
    "Personagens Recorrentes": ["recorrente"],
    "Personagens Secundários": ["secundário"],
}


# ---------------------------------------------------------------------------
# Parse e reconstrução (reutilizando padrão do Silver)
# ---------------------------------------------------------------------------

def parse_md(conteudo: str) -> tuple[dict, str]:
    """Separa o frontmatter YAML do corpo Markdown."""
    frontmatter = {}
    corpo = conteudo

    if conteudo.startswith("---"):
        partes = conteudo.split("---", 2)
        if len(partes) >= 3:
            yaml_bloco = partes[1].strip()
            corpo = partes[2]

            for linha in yaml_bloco.split("\n"):
                linha = linha.strip()
                if ":" in linha:
                    chave, valor = linha.split(":", 1)
                    chave = chave.strip()
                    valor = valor.strip()
                    # Tenta parsear valores especiais
                    if valor.lower() in ("true", "false"):
                        frontmatter[chave] = valor.lower() == "true"
                    else:
                        frontmatter[chave] = valor.strip('"').strip("'")

    return frontmatter, corpo


def escape_yaml_value(valor: str) -> str:
    """Garante que o valor YAML esteja seguro entre aspas duplas."""
    valor = valor.replace('\\', '\\\\').replace('"', '\\"')
    return valor


def reconstruir_md(frontmatter: dict, corpo: str) -> str:
    """Reconstrói o Markdown com frontmatter YAML."""
    linhas = ["---"]
    for chave, valor in frontmatter.items():
        if isinstance(valor, bool):
            linhas.append(f"{chave}: {'true' if valor else 'false'}")
        elif isinstance(valor, (int, float)):
            linhas.append(f"{chave}: {valor}")
        elif isinstance(valor, list):
            if valor:
                items_str = ", ".join(f'"{escape_yaml_value(v)}"' for v in valor)
                linhas.append(f"{chave}: [{items_str}]")
        else:
            valor_escaped = escape_yaml_value(str(valor))
            linhas.append(f'{chave}: "{valor_escaped}"')
    linhas.append("---")
    linhas.append(corpo)
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Enriquecimentos
# ---------------------------------------------------------------------------

def gerar_resumo(corpo: str, max_chars: int = 200) -> str:
    """Extrai o primeiro parágrafo significativo como resumo."""
    linhas = corpo.strip().split("\n")
    for linha in linhas:
        linha_limpa = linha.strip()
        # Pula headers, linhas vazias, linhas de formatação
        if not linha_limpa or linha_limpa.startswith("#") or linha_limpa.startswith("**") or linha_limpa.startswith("-"):
            continue
        # Usa o primeiro parágrafo real
        resumo = linha_limpa
        if len(resumo) > max_chars:
            # Corta no último espaço antes do limite
            resumo = resumo[:max_chars].rsplit(" ", 1)[0] + "..."
        return resumo
    return ""


def extrair_personagens_mencionados(texto: str) -> list[str]:
    """Encontra personagens conhecidos mencionados no texto."""
    mencionados = []
    for personagem in PERSONAGENS_CONHECIDOS:
        if personagem in texto:
            mencionados.append(personagem)
    return sorted(set(mencionados))


def gerar_tags_personagem(frontmatter: dict, corpo: str) -> list[str]:
    """Gera tags semânticas para um personagem."""
    tags = []

    # Tags da categoria
    categoria = frontmatter.get("categoria", "")
    if categoria in TAGS_CATEGORIA:
        tags.extend(TAGS_CATEGORIA[categoria])

    # Tags do conteúdo
    texto = corpo.lower()
    if "vampir" in texto:
        tags.append("vampiro")
    if "mágic" in texto or "magia" in texto or "feitiç" in texto:
        tags.append("mágico")
    if "robot" in texto or "máquina" in texto or "videogame" in texto:
        tags.append("robô")
    if "demon" in texto or "demônio" in texto or "noitosfera" in texto:
        tags.append("demônio")
    if "rei " in texto or "rainha" in texto or "princesa" in texto or "príncipe" in texto:
        tags.append("realeza")
    if "guerra" in texto or "batalha" in texto or "luta" in texto:
        tags.append("combate")
    if "amor" in texto or "namor" in texto or "relacionamento" in texto:
        tags.append("relacionamento")

    return sorted(set(tags))


def gerar_tags_objeto(frontmatter: dict, corpo: str) -> list[str]:
    """Gera tags semânticas para um objeto."""
    tags = []
    nome = frontmatter.get("nome", "").lower()
    texto = corpo.lower()
    tipo = frontmatter.get("tipo_objeto", "").lower()

    if "espada" in nome or "espada" in tipo:
        tags.append("arma")
        tags.append("espada")
    if "poção" in nome or "soro" in nome or "antídoto" in nome:
        tags.append("poção")
    if "livro" in nome or "diário" in nome or "revista" in nome:
        tags.append("livro")
    if "varinha" in nome:
        tags.append("arma")
        tags.append("mágico")
    if "máquina" in nome or "robô" in nome:
        tags.append("tecnologia")
    if "musical" in texto or "violão" in texto or "viola" in texto or "flauta" in texto or "baixo" in nome:
        tags.append("música")
    if "mágic" in texto or "magia" in texto or "encantad" in texto:
        tags.append("mágico")
    if "mochila" in nome or "chapéu" in nome or "camisa" in nome or "armadura" in nome:
        tags.append("vestimenta")

    return sorted(set(tags))


def gerar_tags_lugar(frontmatter: dict, corpo: str) -> list[str]:
    """Gera tags semânticas para um lugar."""
    tags = []
    nome = frontmatter.get("nome", "").lower()
    texto = corpo.lower()

    if "reino" in nome:
        tags.append("reino")
    if "doce" in nome or "doce" in texto[:200]:
        tags.append("reino_doce")
    if "noitosfera" in nome or "noitosfera" in texto[:200]:
        tags.append("noitosfera")
    if "gelado" in nome or "gelo" in nome:
        tags.append("gelo")
    if "floresta" in nome:
        tags.append("natureza")
    if "castelo" in nome or "palácio" in nome:
        tags.append("construção")
    if "masmorra" in nome or "calabouço" in nome:
        tags.append("masmorra")
    if "destruíd" in texto[:500] or "destruída" in texto[:500]:
        tags.append("destruído")

    return sorted(set(tags))


def gerar_tags_episodio(frontmatter: dict, corpo: str) -> list[str]:
    """Gera tags semânticas para um episódio."""
    tags = []
    texto = corpo.lower()

    temporada = frontmatter.get("temporada", "")
    try:
        t = int(temporada)
        tags.append(f"temporada_{t}")
    except (ValueError, TypeError):
        pass

    if "batalha" in texto or "luta" in texto or "guerra" in texto:
        tags.append("ação")
    if "amor" in texto or "namor" in texto or "beij" in texto:
        tags.append("romance")
    if "flashback" in texto or "passado" in texto:
        tags.append("flashback")
    if "morte" in texto or "morr" in texto:
        tags.append("morte")
    if "magia" in texto or "feitiç" in texto or "mágic" in texto:
        tags.append("magia")

    if frontmatter.get("incompleto"):
        tags.append("incompleto")

    return sorted(set(tags))


def adicionar_contexto_rag(frontmatter: dict, corpo: str) -> str:
    """
    Adiciona uma linha de contexto semântico ao início do corpo,
    para melhorar a qualidade do retrieval.
    """
    tipo = frontmatter.get("tipo", "")
    nome = frontmatter.get("nome", "")

    contextos = {
        "personagem": f"Este documento contém informações sobre o personagem {nome} da série Hora de Aventura (Adventure Time).",
        "lugar": f"Este documento descreve o local {nome} do universo de Hora de Aventura (Adventure Time).",
        "objeto": f"Este documento descreve o objeto/item {nome} da série Hora de Aventura (Adventure Time).",
        "episodio": "",  # Episódios já tem contexto suficiente no frontmatter
    }

    contexto = contextos.get(tipo, "")
    if contexto:
        return f"\n\n> {contexto}\n{corpo}"

    return corpo


# ---------------------------------------------------------------------------
# Processador principal
# ---------------------------------------------------------------------------

def processar_arquivo(caminho_silver: str, caminho_gold: str, categoria: str):
    """Processa um único arquivo Silver → Gold."""
    with open(caminho_silver, "r", encoding="utf-8") as f:
        conteudo = f.read()

    frontmatter, corpo = parse_md(conteudo)

    # 1. Gerar resumo
    resumo = gerar_resumo(corpo)
    if resumo:
        frontmatter["resumo"] = resumo

    # 2. Tags semânticas
    if categoria == "characters":
        tags = gerar_tags_personagem(frontmatter, corpo)
    elif categoria == "items":
        tags = gerar_tags_objeto(frontmatter, corpo)
    elif categoria == "places":
        tags = gerar_tags_lugar(frontmatter, corpo)
    elif categoria == "episodes":
        tags = gerar_tags_episodio(frontmatter, corpo)
    else:
        tags = []

    if tags:
        frontmatter["tags"] = tags

    # 3. Cross-referencing: personagens mencionados
    if categoria in ("episodes", "places", "items"):
        mencionados = extrair_personagens_mencionados(corpo)
        if mencionados:
            frontmatter["personagens_mencionados"] = mencionados

    # 4. Contexto RAG
    corpo = adicionar_contexto_rag(frontmatter, corpo)

    # 5. Reconstruir e salvar
    resultado = reconstruir_md(frontmatter, corpo)

    with open(caminho_gold, "w", encoding="utf-8") as f:
        f.write(resultado)


def processar_json(caminho_silver: str, caminho_gold: str):
    """Copia e enriquece o JSON consolidado."""
    with open(caminho_silver, "r", encoding="utf-8") as f:
        dados = json.load(f)

    # Adiciona resumo e tags a cada entrada do JSON
    for item in dados if isinstance(dados, list) else []:
        # Gerar resumo do campo descricao se existir
        descricao = item.get("descricao", "")
        if descricao and not item.get("resumo"):
            if len(descricao) > 200:
                item["resumo"] = descricao[:200].rsplit(" ", 1)[0] + "..."
            else:
                item["resumo"] = descricao

    with open(caminho_gold, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stats = {cat: {"total": 0, "processados": 0, "com_tags": 0, "com_refs": 0} for cat in CATEGORIAS}

    for categoria in CATEGORIAS:
        silver_cat = os.path.join(SILVER_DIR, categoria)
        gold_cat = os.path.join(GOLD_DIR, categoria)
        os.makedirs(gold_cat, exist_ok=True)

        if not os.path.exists(silver_cat):
            print(f"⚠ Silver/{categoria} não encontrado, pulando...")
            continue

        arquivos = sorted(os.listdir(silver_cat))
        total = len(arquivos)
        print(f"\n{'='*60}")
        print(f"Enriquecendo: {categoria} ({total} arquivos)")
        print(f"{'='*60}")

        for i, nome_arquivo in enumerate(arquivos, start=1):
            caminho_silver = os.path.join(silver_cat, nome_arquivo)
            caminho_gold = os.path.join(gold_cat, nome_arquivo)

            if not os.path.isfile(caminho_silver):
                continue

            stats[categoria]["total"] += 1

            try:
                if nome_arquivo.endswith(".json"):
                    processar_json(caminho_silver, caminho_gold)
                    print(f"  [{i:03d}/{total}] ✔ {nome_arquivo} (JSON)")
                elif nome_arquivo.endswith(".md"):
                    processar_arquivo(caminho_silver, caminho_gold, categoria)

                    # Verifica enriquecimentos aplicados
                    with open(caminho_gold, "r", encoding="utf-8") as f:
                        gold_content = f.read()[:1000]

                    tem_tags = "tags:" in gold_content
                    tem_refs = "personagens_mencionados:" in gold_content

                    if tem_tags:
                        stats[categoria]["com_tags"] += 1
                    if tem_refs:
                        stats[categoria]["com_refs"] += 1

                    indicadores = []
                    if tem_tags:
                        indicadores.append("tags")
                    if tem_refs:
                        indicadores.append("refs")
                    info = f" ({', '.join(indicadores)})" if indicadores else ""
                    print(f"  [{i:03d}/{total}] ✔ {nome_arquivo}{info}")
                else:
                    with open(caminho_silver, "rb") as src, open(caminho_gold, "wb") as dst:
                        dst.write(src.read())
                    print(f"  [{i:03d}/{total}] → {nome_arquivo} (copiado)")

                stats[categoria]["processados"] += 1

            except Exception as e:
                print(f"  [{i:03d}/{total}] ✗ {nome_arquivo} ERRO: {e}")

    # Relatório final
    print(f"\n{'='*60}")
    print("RELATÓRIO GOLD")
    print(f"{'='*60}")
    for cat in CATEGORIAS:
        s = stats[cat]
        print(f"  {cat:15s}: {s['processados']:3d}/{s['total']:3d} processados "
              f"| {s['com_tags']:3d} com tags | {s['com_refs']:3d} com cross-refs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
