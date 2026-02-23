"""
scrape_places.py
-----------------
Scraper de alta qualidade para lugares de Hora de Aventura (PT-BR).
Foco em dados limpos e bem estruturados para RAG.

Descobre lugares automaticamente via Categoria:Lugares (com paginação).
Captura TODAS as seções h2 de cada página sem lista fixa de seções.

Saída: knowledge/bronze/places/<Nome>.md
"""

import json
import os
import re
import sys
import time
import unicodedata

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_URL     = "https://horadeaventura.fandom.com"
CATEGORY_URL = f"{BASE_URL}/pt-br/wiki/Categoria:Lugares"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "bronze", "places"
)

# Seções de ruído/navegação a ignorar
SECOES_IGNORAR = {
    "referências", "referencias",
    "ver também", "ver tambem",
    "galeria",
    "notas",
    "ligações externas", "ligacoes externas",
    "navegação", "navegacao",
    "índice", "indice",
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
    """Remove ruído típico de wikis: refs [1], notas, espaços extras."""
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
# Extração da infobox de lugares
# ---------------------------------------------------------------------------

MAPA_LABELS_LUGAR = {
    "localização":     "localizacao",
    "localizacao":     "localizacao",
    "local":           "localizacao",
    "dono":            "dono",
    "proprietário":    "dono",
    "governante":      "governante",
    "líder":           "governante",
    "lider":           "governante",
    "residente":       "residentes",
    "residentes":      "residentes",
    "habitante":       "residentes",
    "habitantes":      "residentes",
    "estado":          "status",
    "status":          "status",
    "tipo":            "tipo_lugar",
    "população":       "populacao",
    "populacao":       "populacao",
    "primeira aparição": "primeira_aparicao",
    "última aparição":   "ultima_aparicao",
}


def extrair_infobox_lugar(soup) -> dict:
    """Extrai campos da infobox de lugares do Fandom."""
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

        chave = MAPA_LABELS_LUGAR.get(label)
        if not chave:
            for k, v in MAPA_LABELS_LUGAR.items():
                if k in label:
                    chave = v
                    break

        if chave and value:
            dados[chave] = value

    return dados

# ---------------------------------------------------------------------------
# Extração genérica de seções h2
# ---------------------------------------------------------------------------

def _extrair_conteudo_bloco(html_bloco: str) -> str:
    """
    Dado HTML bruto de uma seção, extrai h3, h4, parágrafos e listas
    em qualquer nível de aninhamento.
    """
    soup = BeautifulSoup(html_bloco, "html.parser")
    blocos = []
    vistos = set()

    for el in soup.find_all(["h3", "h4", "p", "ul", "ol"]):
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

        if not titulo or titulo.lower() in SECOES_IGNORAR:
            continue

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
    """Extrai parágrafos introdutórios antes do primeiro h2."""
    paragrafos = []
    for child in content.children:
        if child.name == "h2":
            break
        if child.name == "p":
            txt = limpar_texto(child.get_text(separator=" ", strip=True))
            if txt and len(txt) > 30:
                paragrafos.append(txt)
    return "\n\n".join(paragrafos)

# ---------------------------------------------------------------------------
# Descoberta de lugares via categoria (com paginação)
# ---------------------------------------------------------------------------

def navegar(page: Page, url: str) -> BeautifulSoup:
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector(".mw-parser-output", timeout=20000)
    except Exception:
        pass
    return BeautifulSoup(page.content(), "html.parser")


def scrape_links_categoria(page: Page) -> list[dict]:
    """
    Navega pela Categoria:Lugares com paginação e coleta todos os links.
    Ignora subcategorias.
    """
    print(f"Buscando links: {CATEGORY_URL}")
    todos = []
    url = CATEGORY_URL

    while url:
        soup = navegar(page, url)
        members = soup.find("div", class_="category-page__members")
        if not members:
            break

        for item in members.find_all("a", class_="category-page__member-link", href=True):
            href = item["href"]
            nome = item.get_text(strip=True)
            if "Categoria:" in href or "Categoria:" in nome:
                continue
            link = BASE_URL + href if href.startswith("/") else href
            todos.append({"nome": nome, "url": link})

        next_btn = soup.find("a", class_="category-page__pagination-next")
        url = None
        if next_btn and next_btn.get("href"):
            next_href = next_btn["href"]
            url = BASE_URL + next_href if next_href.startswith("/") else next_href

    print(f"  → {len(todos)} lugares encontrados\n")
    return todos

# ---------------------------------------------------------------------------
# Scraping de um lugar
# ---------------------------------------------------------------------------

def scrape_lugar(page: Page, meta: dict) -> dict:
    """Visita a página do lugar e extrai infobox + todas as seções."""
    url = meta["url"]
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector(".mw-parser-output", timeout=15000)
    except Exception:
        pass

    soup    = BeautifulSoup(page.content(), "html.parser")
    content = soup.find("div", class_="mw-parser-output")

    resultado = {
        "nome":    meta["nome"],
        "url":     url,
        "infobox": {},
        "intro":   "",
        "secoes":  {},
    }

    if not content:
        return resultado

    resultado["infobox"] = extrair_infobox_lugar(soup)
    resultado["intro"]   = extrair_intro(content)
    resultado["secoes"]  = extrair_todas_secoes(content)
    return resultado

# ---------------------------------------------------------------------------
# Geração do Markdown RAG-ready
# ---------------------------------------------------------------------------

def gerar_markdown(lugar: dict) -> str:
    nome    = lugar["nome"]
    infobox = lugar.get("infobox", {})
    intro   = lugar.get("intro", "")
    secoes  = lugar.get("secoes", {})

    linhas = ["---"]
    linhas.append('tipo: "lugar"')
    linhas.append(f'nome: "{nome}"')

    for campo in ["tipo_lugar", "localizacao", "governante", "dono",
                  "residentes", "populacao", "status",
                  "primeira_aparicao", "ultima_aparicao"]:
        val = infobox.get(campo, "")
        if val:
            linhas.append(f'{campo}: "{val}"')

    linhas.append(f'url: "{lugar["url"]}"')
    linhas.append("---")
    linhas.append("")

    # Frase âncora para o embedding
    linhas.append(f"> {nome} é um lugar de Hora de Aventura.")
    linhas.append("")
    linhas.append(f"# {nome}")
    linhas.append("")

    # Introdução
    if intro:
        linhas.append("## Descrição")
        linhas.append(intro)
        linhas.append("")

    # Todas as seções descobertas automaticamente, na ordem da página
    for chave, dados in secoes.items():
        linhas.append(f"## {dados['titulo']}")
        linhas.append(dados["conteudo"])
        linhas.append("")

    return "\n".join(linhas)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    todos = []
    erros = []

    print(f"\n{'='*60}")
    print(f"Scraping lugares para RAG")
    print(f"Saída: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page()

        # 1. Descobre todos os lugares via categoria
        lugares_meta = scrape_links_categoria(page)
        total = len(lugares_meta)

        # 2. Scrape individual de cada lugar
        for i, meta in enumerate(lugares_meta, 1):
            print(f"[{i:03d}/{total}] {meta['nome']}")
            try:
                dados = scrape_lugar(page, meta)
                todos.append(dados)

                md      = gerar_markdown(dados)
                arquivo = os.path.join(OUTPUT_DIR, f"{slugify(meta['nome'])}.md")
                with open(arquivo, "w", encoding="utf-8") as f:
                    f.write(md)

                n_secoes = len(dados.get("secoes", {}))
                n_chars  = len(md)
                secoes_nomes = ", ".join(
                    dados["secoes"][k]["titulo"] for k in dados["secoes"]
                ) or "(sem seções)"
                print(f"  ✔ {n_secoes} seções | {n_chars} chars")
                print(f"     └─ {secoes_nomes}")

            except Exception as e:
                print(f"  ✗ ERRO: {e}")
                erros.append(meta["nome"])
                todos.append({**meta, "erro": str(e)})

            time.sleep(0.5)

        browser.close()

    # 3. JSON consolidado
    json_path = os.path.join(OUTPUT_DIR, "lugares_rag.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✔ Processados : {len(todos) - len(erros)}/{total}")
    print(f"✗ Erros       : {len(erros)} {erros if erros else ''}")
    print(f"JSON          : {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()