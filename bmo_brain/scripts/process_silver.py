"""
process_silver.py
-----------------
Camada Silver do pipeline de qualidade de dados do BMO Brain.
Lê os Markdowns brutos de knowledge/bronze/ e aplica limpezas e
normalizações, salvando o resultado em knowledge/silver/.

Transformações aplicadas:
  1. Limpar referências wiki residuais ([1], [2], etc.)
  2. Normalizar espaços antes de pontuação (Finn . → Finn.)
  3. Limpar ruído da infobox (vírgulas soltas, aspas vazias, etc.)
  4. Corrigir numero_ep dos episódios (extrair do nome do arquivo)
  5. Separar personagens secundários em lista
  6. Marcar arquivos vazios/stub com incompleto: true
  7. Remover duplicação de conteúdo entre seções
  8. Normalizar YAML frontmatter
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
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "silver")

CATEGORIAS = ["episodes", "characters", "places", "items"]

# Tamanho mínimo (bytes) para considerar um episódio como "completo"
MIN_EPISODE_SIZE = 400


# ---------------------------------------------------------------------------
# Funções de limpeza de texto
# ---------------------------------------------------------------------------

def limpar_refs_wiki(texto: str) -> str:
    """Remove referências wiki como [1], [2], etc."""
    return re.sub(r"\s*\[\d+\]", "", texto)


def normalizar_espacos_pontuacao(texto: str) -> str:
    """Corrige espaços antes de pontuação: 'Finn .' → 'Finn.'"""
    # Espaço antes de pontuação
    texto = re.sub(r"\s+([.,;:!?\)])", r"\1", texto)
    # Espaço depois de abertura de parêntese
    texto = re.sub(r"\(\s+", "(", texto)
    # Múltiplos espaços
    texto = re.sub(r"  +", " ", texto)
    return texto


def limpar_valor_infobox(valor: str) -> str:
    """
    Limpa valores sujos da infobox.
    Ex: '"", Memória de Uma Memória, ""' → 'Memória de Uma Memória'
    Ex: 'Finn, e, Jake, (Antigamente)' → 'Finn e Jake (Antigamente)'
    Ex: 'Reino Doce, ,, Ooo, [, 1, ]' → 'Reino Doce, Ooo'
    """
    # Remove aspas duplas vazias
    valor = re.sub(r'""', "", valor)

    # Remove referências [1], [2], etc.
    valor = re.sub(r"\[[\s,]*\d+[\s,]*\]", "", valor)

    # Remove colchetes soltos com conteúdo numérico
    valor = re.sub(r"\[\s*\]", "", valor)

    # Limpa labels embutidos nos valores (ex: 'Antigo dono:, Marceline')
    # Transforma 'Label:, Valor' em 'Label: Valor'
    valor = re.sub(r":\s*,\s*", ": ", valor)

    # Limpa vírgulas soltas e múltiplas
    valor = re.sub(r",\s*,+", ",", valor)           # ,, → ,
    valor = re.sub(r"^\s*,\s*", "", valor)           # vírgula no início
    valor = re.sub(r"\s*,\s*$", "", valor)           # vírgula no fim
    valor = re.sub(r"\s*,\s*", ", ", valor)          # normaliza espaço ao redor

    # Reconstrói frases onde cada palavra foi separada por vírgula
    # Detecta padrão: "palavra, e, palavra" ou "palavra, de, palavra"
    valor = re.sub(r",\s+(e|de|do|da|dos|das|o|a|os|as|no|na|nos|nas|em|por|para|com|até|ou)\s*,\s+", r" \1 ", valor)

    # Remove parênteses com espaço extra dentro
    valor = re.sub(r"\(\s+", "(", valor)
    valor = re.sub(r"\s+\)", ")", valor)

    # Limpa espaços múltiplos
    valor = re.sub(r"  +", " ", valor)
    valor = valor.strip()

    return valor


def escape_yaml_value(valor: str) -> str:
    """Garante que o valor YAML esteja seguro entre aspas duplas."""
    # Escapa aspas duplas internas
    valor = valor.replace('\\', '\\\\').replace('"', '\\"')
    return valor


# ---------------------------------------------------------------------------
# Parse e reconstrução de Markdown com frontmatter
# ---------------------------------------------------------------------------

def parse_md(conteudo: str) -> tuple[dict, str]:
    """
    Separa o frontmatter YAML do corpo Markdown.
    Retorna (dict_frontmatter, corpo_markdown).
    """
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
                    valor = valor.strip().strip('"').strip("'")
                    frontmatter[chave] = valor

    return frontmatter, corpo


def reconstruir_md(frontmatter: dict, corpo: str) -> str:
    """Reconstrói o Markdown com frontmatter YAML limpo."""
    linhas = ["---"]
    for chave, valor in frontmatter.items():
        if isinstance(valor, (int, float)):
            linhas.append(f"{chave}: {valor}")
        elif isinstance(valor, bool):
            linhas.append(f"{chave}: {'true' if valor else 'false'}")
        else:
            valor_escaped = escape_yaml_value(str(valor))
            linhas.append(f'{chave}: "{valor_escaped}"')
    linhas.append("---")
    linhas.append(corpo)
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Processadores por categoria
# ---------------------------------------------------------------------------

def extrair_numero_ep_do_arquivo(nome_arquivo: str) -> int | None:
    """
    Extrai o número correto do episódio do nome do arquivo.
    Ex: 'T09_E2254_Elementos.md' → episódio depende da temporada
    Ex: 'T01_E001_Panico...' → 1
    """
    match = re.match(r"T(\d+)_E(\d+)", nome_arquivo)
    if match:
        num_bruto = int(match.group(2))
        # Se o número parece ser concatenado (>200), é provavelmente
        # o número real + número acumulado. Pega os primeiros 1-2 dígitos
        # baseado no padrão: TXX_EYYZZZ onde YY=num_ep e ZZZ=acumulado
        if num_bruto > 200:
            # Para temporadas com 2 dígitos de ep (ex: E1062 = ep 10, acumulado 62)
            s = str(num_bruto)
            if len(s) == 4:
                return int(s[:2])
            elif len(s) == 3:
                return int(s[:1])
        return num_bruto
    return None


def processar_episodio(frontmatter: dict, corpo: str, nome_arquivo: str, tamanho: int) -> tuple[dict, str]:
    """Aplica limpezas específicas para episódios."""

    # Corrigir numero_ep se parecer errado
    num_ep = frontmatter.get("numero_ep")
    if num_ep:
        try:
            num_ep_int = int(num_ep)
            if num_ep_int > 200:
                corrigido = extrair_numero_ep_do_arquivo(nome_arquivo)
                if corrigido is not None:
                    frontmatter["numero_ep"] = corrigido
        except ValueError:
            pass

    # Marcar episódios incompletos
    if tamanho < MIN_EPISODE_SIZE:
        frontmatter["incompleto"] = True

    # Separar personagens secundários em lista (se for uma string longa)
    corpo = separar_personagens_secundarios(corpo)

    return frontmatter, corpo


def separar_personagens_secundarios(corpo: str) -> str:
    """
    Detecta se os personagens secundários estão numa string única
    e tenta separá-los em itens de lista.
    """
    linhas = corpo.split("\n")
    novas_linhas = []
    dentro_secundarios = False

    for i, linha in enumerate(linhas):
        if "### Secundários" in linha or "### Secundarios" in linha:
            dentro_secundarios = True
            novas_linhas.append(linha)
            continue

        if dentro_secundarios and linha.startswith("- "):
            # Verifica se é uma linha enorme com muitos nomes juntos
            conteudo = linha[2:].strip()
            if len(conteudo) > 100 and conteudo.count(" ") > 10 and "," not in conteudo:
                # Provavelmente nomes grudados — tenta separar por maiúsculas
                # Padrão: "Povo Doce Moranguinha Chico Banana..."
                # Infelizmente sem delimitador claro, mantém como está
                # mas prefixamos uma nota
                novas_linhas.append(linha)
            else:
                novas_linhas.append(linha)
            dentro_secundarios = False
            continue

        if dentro_secundarios and linha.strip() and not linha.startswith("#"):
            dentro_secundarios = False

        novas_linhas.append(linha)

    return "\n".join(novas_linhas)


def processar_personagem(frontmatter: dict, corpo: str) -> tuple[dict, str]:
    """Aplica limpezas específicas para personagens."""

    # Detectar e remover duplicação de conteúdo
    corpo = remover_secoes_duplicadas(corpo)

    return frontmatter, corpo


def remover_secoes_duplicadas(corpo: str) -> str:
    """
    Se duas seções possuem conteúdo idêntico ou uma contém a outra,
    remove o duplicado (mantém a versão mais longa).
    Ex: Aparência e Habilidades com mesmo texto → mantém só uma.
    """
    secoes = extrair_secoes(corpo)
    secoes_para_remover = []
    nomes = list(secoes.keys())

    for i, nome_a in enumerate(nomes):
        conteudo_a = secoes[nome_a].strip()
        if not conteudo_a or nome_a in secoes_para_remover:
            continue
        for j, nome_b in enumerate(nomes):
            if i >= j or nome_b in secoes_para_remover:
                continue
            conteudo_b = secoes[nome_b].strip()
            if not conteudo_b:
                continue

            # Duplicação exata
            if conteudo_a == conteudo_b:
                secoes_para_remover.append(nome_b)
            # Conteúdo B está inteiramente contido em A → remove B
            elif conteudo_b in conteudo_a:
                secoes_para_remover.append(nome_b)
            # Conteúdo A está inteiramente contido em B → remove A
            elif conteudo_a in conteudo_b:
                secoes_para_remover.append(nome_a)

    # Remove seções duplicadas do corpo
    for secao in secoes_para_remover:
        # Remove a seção ## Nome + conteúdo até a próxima seção ##
        padrao = re.compile(
            rf"^## {re.escape(secao)}\n.*?(?=^## |\Z)",
            re.MULTILINE | re.DOTALL
        )
        corpo = padrao.sub("", corpo)

    return corpo


def extrair_secoes(corpo: str) -> dict[str, str]:
    """Extrai seções h2 do corpo Markdown."""
    secoes = {}
    secao_atual = None
    conteudo_atual = []

    for linha in corpo.split("\n"):
        match = re.match(r"^## (.+)$", linha)
        if match:
            if secao_atual:
                secoes[secao_atual] = "\n".join(conteudo_atual)
            secao_atual = match.group(1).strip()
            conteudo_atual = []
        elif secao_atual:
            conteudo_atual.append(linha)

    if secao_atual:
        secoes[secao_atual] = "\n".join(conteudo_atual)

    return secoes


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def limpar_texto_geral(texto: str) -> str:
    """Aplica limpezas gerais ao texto do corpo."""
    texto = limpar_refs_wiki(texto)
    texto = normalizar_espacos_pontuacao(texto)
    return texto


def limpar_frontmatter_valores(frontmatter: dict) -> dict:
    """Limpa todos os valores string do frontmatter."""
    campos_infobox = [
        "localizacao", "dono", "residentes", "estado_atual", "governante",
        "populacao", "primeira_aparicao", "ultima_aparicao", "tipo_lugar",
        "tipo_objeto", "criador", "material", "poderes", "estado",
        "nome_completo", "apelidos", "genero", "idade", "especie",
        "ocupacao", "residencia", "parentes", "animal_estimacao",
        "dublador", "nome_original", "diretor", "roteiro", "storyboard",
        "audiencia", "codigo_producao", "data_exibicao", "data_exibicao_br",
    ]
    for chave in campos_infobox:
        if chave in frontmatter and isinstance(frontmatter[chave], str):
            frontmatter[chave] = limpar_valor_infobox(frontmatter[chave])

    return frontmatter


def processar_arquivo(
    caminho_bronze: str,
    caminho_silver: str,
    categoria: str,
    nome_arquivo: str,
):
    """Processa um único arquivo da Bronze → Silver."""
    with open(caminho_bronze, "r", encoding="utf-8") as f:
        conteudo = f.read()

    tamanho = len(conteudo.encode("utf-8"))

    # 1. Parse
    frontmatter, corpo = parse_md(conteudo)

    # 2. Limpezas gerais
    frontmatter = limpar_frontmatter_valores(frontmatter)
    corpo = limpar_texto_geral(corpo)

    # 3. Limpezas específicas por categoria
    if categoria == "episodes":
        frontmatter, corpo = processar_episodio(frontmatter, corpo, nome_arquivo, tamanho)
    elif categoria == "characters":
        frontmatter, corpo = processar_personagem(frontmatter, corpo)

    # 4. Reconstruir e salvar
    resultado = reconstruir_md(frontmatter, corpo)

    with open(caminho_silver, "w", encoding="utf-8") as f:
        f.write(resultado)


def processar_json(caminho_bronze: str, caminho_silver: str):
    """Processa um arquivo JSON: aplica limpeza nos valores string."""
    with open(caminho_bronze, "r", encoding="utf-8") as f:
        dados = json.load(f)

    def limpar_valor_recursivo(obj):
        if isinstance(obj, str):
            obj = limpar_refs_wiki(obj)
            obj = normalizar_espacos_pontuacao(obj)
            obj = limpar_valor_infobox(obj)
            return obj
        elif isinstance(obj, dict):
            return {k: limpar_valor_recursivo(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [limpar_valor_recursivo(item) for item in obj]
        return obj

    dados_limpos = limpar_valor_recursivo(dados)

    with open(caminho_silver, "w", encoding="utf-8") as f:
        json.dump(dados_limpos, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stats = {cat: {"total": 0, "processados": 0, "incompletos": 0} for cat in CATEGORIAS}

    for categoria in CATEGORIAS:
        bronze_cat = os.path.join(BRONZE_DIR, categoria)
        silver_cat = os.path.join(SILVER_DIR, categoria)
        os.makedirs(silver_cat, exist_ok=True)

        if not os.path.exists(bronze_cat):
            print(f"⚠ Bronze/{categoria} não encontrado, pulando...")
            continue

        arquivos = sorted(os.listdir(bronze_cat))
        total = len(arquivos)
        print(f"\n{'='*60}")
        print(f"Processando: {categoria} ({total} arquivos)")
        print(f"{'='*60}")

        for i, nome_arquivo in enumerate(arquivos, start=1):
            caminho_bronze = os.path.join(bronze_cat, nome_arquivo)
            caminho_silver = os.path.join(silver_cat, nome_arquivo)

            if not os.path.isfile(caminho_bronze):
                continue

            stats[categoria]["total"] += 1

            try:
                if nome_arquivo.endswith(".json"):
                    processar_json(caminho_bronze, caminho_silver)
                    print(f"  [{i:03d}/{total}] ✔ {nome_arquivo} (JSON)")
                elif nome_arquivo.endswith(".md"):
                    processar_arquivo(caminho_bronze, caminho_silver, categoria, nome_arquivo)

                    # Verifica se foi marcado como incompleto
                    with open(caminho_silver, "r", encoding="utf-8") as f:
                        if "incompleto" in f.read()[:500]:
                            stats[categoria]["incompletos"] += 1
                            print(f"  [{i:03d}/{total}] ⚠ {nome_arquivo} (incompleto)")
                        else:
                            print(f"  [{i:03d}/{total}] ✔ {nome_arquivo}")
                else:
                    # Copia outros arquivos sem processamento
                    with open(caminho_bronze, "rb") as src, open(caminho_silver, "wb") as dst:
                        dst.write(src.read())
                    print(f"  [{i:03d}/{total}] → {nome_arquivo} (copiado)")

                stats[categoria]["processados"] += 1

            except Exception as e:
                print(f"  [{i:03d}/{total}] ✗ {nome_arquivo} ERRO: {e}")

    # Relatório final
    print(f"\n{'='*60}")
    print("RELATÓRIO SILVER")
    print(f"{'='*60}")
    for cat in CATEGORIAS:
        s = stats[cat]
        inc = f" ({s['incompletos']} incompletos)" if s["incompletos"] else ""
        print(f"  {cat:15s}: {s['processados']:3d}/{s['total']:3d} processados{inc}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
