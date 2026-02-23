"""
scrape_personagens_rag.py
--------------------------
Scraper de alta qualidade para personagens de Hora de Aventura (PT-BR).
Foco em dados limpos e bem estruturados para RAG.

Captura TODAS as seções h2 da página automaticamente — sem lista fixa.
Saída: knowledge/bronze/characters/<Nome>.md
"""

import json
import os
import re
import sys
import time
import unicodedata
from urllib.parse import quote

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_URL  = "https://horadeaventura.fandom.com"
BASE_PAGE = f"{BASE_URL}/pt-br/wiki/"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "bronze", "characters"
)

# Personagens a scraper — (nome exibição, slug wiki, categoria)
PERSONAGENS_RAG = [
    ("Finn",             "Finn",              "Protagonistas"),
    ("Jake",             "Jake",              "Protagonistas"),
    ("BMO",              "BMO",               "Protagonistas"),
    ("Princesa Jujuba",  "Princesa_Jujuba",   "Núcleo principal"),
    ("Marceline",        "Marceline",         "Núcleo principal"),
    ("Rei Gelado",       "Rei_Gelado",        "Núcleo principal"),
    ("Gunter",           "Gunter",            "Núcleo principal"),
    ("Princesa de Fogo", "Princesa_de_Fogo",  "Reino de Fogo"),
    ("Lich",             "Lich",              "Vilões principais"),
]

# Seções de navegação/referência que devem ser ignoradas
SECOES_IGNORAR = {
    "referências", "referencias",
    "ver também", "ver tambem",
    "galeria",
    "notas",
    "ligações externas", "ligacoes externas",
    "navegação", "navegacao",
    "índice", "indice",
    "conteúdo", "conteudo",
}

# ---------------------------------------------------------------------------
# Utilitários de texto
# ---------------------------------------------------------------------------

def slugify(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s-]", "", texto).strip()
    return re.sub(r"[\s]+", "_", texto)[:80]


def limpar_texto(txt: str) -> str:
    """Remove ruído típico de wikis: refs [1], notas de rodapé, espaços extras."""
    txt = re.sub(r"\[\d+\]", "", txt)
    txt = re.sub(r"\[nota \d+\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\[carece de fontes\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\[\s*(editar|edit)\s*\]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r" {2,}", " ", txt)
    txt = "\n".join(
        l for l in txt.splitlines()
        if l.strip() and not re.fullmatch(r"[-=_*]{3,}", l.strip())
    )
    return txt.strip()

# ---------------------------------------------------------------------------
# Extração da infobox
# ---------------------------------------------------------------------------

MAPA_LABELS = {
    "nome completo":     "nome_completo",
    "apelido":           "apelidos",
    "alcunha":           "apelidos",
    "sexo":              "genero",
    "gênero":            "genero",
    "genero":            "genero",
    "idade":             "idade",
    "espécie":           "especie",
    "especie":           "especie",
    "raça":              "especie",
    "ocupação":          "ocupacao",
    "ocupacao":          "ocupacao",
    "profissão":         "ocupacao",
    "residência":        "residencia",
    "residencia":        "residencia",
    "lar":               "residencia",
    "parentes":          "parentes",
    "família":           "parentes",
    "familia":           "parentes",
    "estado":            "status",
    "status":            "status",
    "primeira aparição": "primeira_aparicao",
    "última aparição":   "ultima_aparicao",
    "voz original":      "voz_original",
    "dublador":          "voz_dublagem",
    "dublagem":          "voz_dublagem",
}


def extrair_infobox(soup) -> dict:
    dados = {}
    infobox = soup.find("aside", class_="portable-infobox")
    if not infobox:
        return dados

    for item in infobox.find_all("div", class_="pi-item"):
        label_tag = item.find("h3", class_="pi-data-label")
        value_tag = item.find("div", class_="pi-data-value")
        if not (label_tag and value_tag):
            continue

        label = label_tag.get_text(strip=True).lower().strip(":")
        value = limpar_texto(value_tag.get_text(separator=", ", strip=True))
        value = re.sub(r",\s*,+", ",", value)
        value = re.sub(r"^,\s*|,\s*$", "", value).strip()

        chave = MAPA_LABELS.get(label)
        if chave and value:
            dados[chave] = value

    return dados

# ---------------------------------------------------------------------------
# Extração genérica de todas as seções h2
# ---------------------------------------------------------------------------

def _extrair_conteudo_bloco(html_bloco: str) -> str:
    """
    Dado um bloco de HTML (conteúdo entre dois h2), extrai todo o texto
    estruturado: h3, h4, parágrafos e listas — em qualquer nível de aninhamento.
    """
    soup = BeautifulSoup(html_bloco, "html.parser")
    blocos = []
    vistos = set()

    for el in soup.find_all(["h3", "h4", "p", "ul", "ol"]):
        # Ignora elementos dentro de aside, figure, table (imagens, navboxes)
        if el.find_parent(["aside", "figure", "table"]):
            continue

        if el.name == "h3":
            span = el.find("span", class_="mw-headline")
            txt = (span or el).get_text(strip=True)
            if txt and txt not in vistos:
                blocos.append(f"\n### {txt}")
                vistos.add(txt)

        elif el.name == "h4":
            span = el.find("span", class_="mw-headline")
            txt = (span or el).get_text(strip=True)
            if txt and txt not in vistos:
                blocos.append(f"\n#### {txt}")
                vistos.add(txt)

        elif el.name == "p":
            txt = limpar_texto(el.get_text(separator=" ", strip=True))
            # Descarta legendas de imagem (muito curtas)
            if txt and len(txt) > 30 and txt not in vistos:
                blocos.append(txt)
                vistos.add(txt)

        elif el.name in ("ul", "ol"):
            for li in el.find_all("li", recursive=False):
                txt = limpar_texto(li.get_text(separator=" ", strip=True))
                if txt and txt not in vistos:
                    blocos.append(f"- {txt}")
                    vistos.add(txt)

    return "\n\n".join(b for b in blocos if b.strip())


def extrair_todas_secoes(content) -> dict:
    """
    Descobre e extrai TODAS as seções h2 da página automaticamente.
    Retorna dict ordenado: {slug_secao: {"titulo": str, "conteudo": str}}
    """
    secoes = {}

    for h2 in content.find_all("h2"):
        span = h2.find("span", class_="mw-headline")
        titulo = (span or h2).get_text(strip=True)

        if not titulo:
            continue

        # Ignora seções de navegação/referência
        if titulo.lower() in SECOES_IGNORAR:
            continue

        # Coleta HTML bruto de tudo até o próximo h2
        html_bloco = []
        for sibling in h2.next_siblings:
            if sibling.name == "h2":
                break
            html_bloco.append(str(sibling))

        if not html_bloco:
            continue

        conteudo = _extrair_conteudo_bloco("".join(html_bloco))
        if conteudo and len(conteudo) > 20:
            chave = slugify(titulo).lower()
            secoes[chave] = {"titulo": titulo, "conteudo": conteudo}

    return secoes


def extrair_intro(content) -> str:
    """Extrai parágrafos introdutórios (antes do primeiro h2)."""
    paragrafos = []
    for child in content.children:
        if child.name == "h2":
            break
        if child.name == "p":
            txt = limpar_texto(child.get_text(separator=" ", strip=True))
            if txt and len(txt) > 30:
                paragrafos.append(txt)
    return "\n\n".join(paragrafos)


def extrair_citacoes(secoes: dict) -> list[str]:
    """Extrai citações da seção de citações, se existir."""
    for chave, dados in secoes.items():
        if "cita" in chave or "frase" in chave:
            linhas = dados["conteudo"].splitlines()
            citacoes = [l.lstrip("- ").strip() for l in linhas if l.strip()]
            return citacoes[:5]
    return []

# ---------------------------------------------------------------------------
# Scraping de um personagem
# ---------------------------------------------------------------------------

def scrape_personagem(page: Page, nome: str, wiki_slug: str, categoria: str) -> dict:
    url = BASE_PAGE + quote(wiki_slug, safe="/_().")
    print(f"  → {url}")

    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector(".mw-parser-output", timeout=15000)
    except Exception:
        pass

    soup = BeautifulSoup(page.content(), "html.parser")
    content = soup.find("div", class_="mw-parser-output")

    resultado = {
        "nome": nome,
        "categoria": categoria,
        "url": url,
        "infobox": {},
        "intro": "",
        "secoes": {},
        "citacoes": [],
    }

    if not content:
        return resultado

    resultado["infobox"] = extrair_infobox(soup)
    resultado["intro"]   = extrair_intro(content)
    resultado["secoes"]  = extrair_todas_secoes(content)
    resultado["citacoes"] = extrair_citacoes(resultado["secoes"])

    return resultado

# ---------------------------------------------------------------------------
# Geração do Markdown RAG-ready
# ---------------------------------------------------------------------------

def gerar_markdown(p: dict) -> str:
    nome      = p["nome"]
    categoria = p["categoria"]
    infobox   = p.get("infobox", {})
    intro     = p.get("intro", "")
    secoes    = p.get("secoes", {})
    citacoes  = p.get("citacoes", [])

    linhas = ["---"]
    linhas.append('tipo: "personagem"')
    linhas.append(f'nome: "{nome}"')
    linhas.append(f'categoria: "{categoria}"')

    for campo in ["nome_completo", "apelidos", "genero", "idade", "especie",
                  "ocupacao", "residencia", "parentes", "status",
                  "primeira_aparicao", "ultima_aparicao", "voz_original", "voz_dublagem"]:
        val = infobox.get(campo, "")
        if val:
            linhas.append(f'{campo}: "{val}"')

    linhas.append(f'url: "{p["url"]}"')
    linhas.append("---")
    linhas.append("")

    # Frase âncora para o embedding
    linhas.append(f"> {nome} é um personagem de Hora de Aventura. Categoria: {categoria}.")
    linhas.append("")
    linhas.append(f"# {nome}")
    linhas.append("")

    # Introdução
    if intro:
        linhas.append("## Descrição")
        linhas.append(intro)
        linhas.append("")

    # Todas as seções descobertas automaticamente, na ordem da página
    chaves_citacao = {k for k in secoes if "cita" in k or "frase" in k}
    for chave, dados in secoes.items():
        if chave in chaves_citacao:
            continue  # citações vão no bloco separado no final
        linhas.append(f"## {dados['titulo']}")
        linhas.append(dados["conteudo"])
        linhas.append("")

    # Citações
    if citacoes:
        linhas.append("## Citações Notáveis")
        for c in citacoes:
            linhas.append(f'> "{c}"')
            linhas.append("")

    return "\n".join(linhas)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(PERSONAGENS_RAG)
    todos = []
    erros = []

    print(f"\n{'='*60}")
    print(f"Scraping {total} personagens para RAG")
    print(f"Saída: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        for i, (nome, slug, categoria) in enumerate(PERSONAGENS_RAG, 1):
            print(f"[{i:02d}/{total}] {nome}")
            try:
                dados = scrape_personagem(page, nome, slug, categoria)
                todos.append(dados)

                md = gerar_markdown(dados)
                arquivo = os.path.join(OUTPUT_DIR, f"{slugify(nome)}.md")
                with open(arquivo, "w", encoding="utf-8") as f:
                    f.write(md)

                n_secoes = len(dados.get("secoes", {}))
                n_chars  = len(md)
                secoes_nomes = ", ".join(
                    dados["secoes"][k]["titulo"] for k in dados["secoes"]
                )
                print(f"  ✔ {n_secoes} seções | {n_chars} chars")
                print(f"     └─ {secoes_nomes}")

            except Exception as e:
                print(f"  ✗ ERRO: {e}")
                erros.append(nome)
                todos.append({"nome": nome, "categoria": categoria, "url": "", "erro": str(e)})

            time.sleep(0.8)

        browser.close()

    json_path = os.path.join(OUTPUT_DIR, "personagens_rag.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✔ Processados : {len(todos) - len(erros)}/{total}")
    print(f"✗ Erros       : {len(erros)} {erros if erros else ''}")
    print(f"JSON          : {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
