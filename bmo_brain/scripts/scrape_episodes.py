"""
scrape_episodes.py
-------------------
Scraper de alta qualidade para episódios de Hora de Aventura (PT-BR).
Foco em dados limpos e bem estruturados para RAG.

Descoberta: lê a Lista_de_Episódios e extrai episódios por temporada via tabelas wikitable.
Conteúdo: captura TODAS as seções h2 de cada página sem lista fixa de seções.

Saída: knowledge/bronze/episodes/T{ss}_E{eee}_{Nome}.md
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

BASE_URL = "https://horadeaventura.fandom.com"
LIST_URL = f"{BASE_URL}/pt-br/wiki/Lista_de_Epis%C3%B3dios"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "bronze", "episodes"
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
# Extração da infobox de episódios
# ---------------------------------------------------------------------------

MAPA_LABELS_EP = {
    "nome original":    "nome_original",
    "original":         "nome_original",
    "produção":         "codigo_producao",
    "código":           "codigo_producao",
    "diretor":          "diretor",
    "roteiro":          "roteiro",
    "história":         "roteiro",
    "storyboard":       "storyboard",
    "audiência":        "audiencia",
    "audiencia":        "audiencia",
}


def extrair_infobox_episodio(soup) -> dict:
    """Extrai campos da infobox específicos de episódios."""
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
        value = limpar_texto(value_tag.get_text(separator=" ", strip=True))

        # Campos de data com distinção Brasil/EUA
        if "exibição" in label and "brasil" in label:
            dados["data_exibicao_br"] = value
            continue
        if "exibição" in label or "estreia" in label:
            dados["data_exibicao"] = value
            continue

        chave = MAPA_LABELS_EP.get(label)
        if not chave:
            for k, v in MAPA_LABELS_EP.items():
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
# Descoberta de episódios via Lista_de_Episódios (lógica única de episódios)
# ---------------------------------------------------------------------------

def navegar(page: Page, url: str) -> BeautifulSoup:
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector(".mw-parser-output", timeout=20000)
    except Exception:
        pass
    return BeautifulSoup(page.content(), "html.parser")


def extrair_numero_temporada(texto_header: str) -> int | None:
    """Extrai número da temporada do texto do header h2."""
    match = re.search(r"(\d+)[ªº°]?\s*[Tt]emporada", texto_header)
    if match:
        return int(match.group(1))
    if "Piloto" in texto_header:
        return 0
    return None


def scrape_lista_episodios(page: Page) -> list[dict]:
    """
    Lê Lista_de_Episódios e extrai metadados de todos os episódios
    a partir das tabelas wikitable organizadas por temporada.
    """
    print(f"Buscando lista de episódios: {LIST_URL}")
    soup = navegar(page, LIST_URL)
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        raise RuntimeError("Não foi possível encontrar o conteúdo da lista.")

    episodios = []
    temporada_atual = None

    for elemento in content.children:
        # Detecta mudança de temporada
        if elemento.name == "h2":
            span = elemento.find("span", class_="mw-headline")
            if span:
                num = extrair_numero_temporada(span.get_text(strip=True))
                if num is not None:
                    temporada_atual = num
                    print(f"  → Temporada {temporada_atual}")

        # Processa tabela de episódios da temporada atual
        elif elemento.name == "table" and "wikitable" in elemento.get("class", []):
            if temporada_atual is None:
                continue

            linhas_dados = [
                tr for tr in elemento.find_all("tr")
                if tr.find("td")
                and not tr.find("th")
                and not tr.find("td", attrs={"colspan": True})
                and len(tr.find_all("td")) >= 3
            ]

            for linha in linhas_dados:
                colunas = linha.find_all("td")
                try:
                    num_ep = int(colunas[0].get_text(strip=True))
                except ValueError:
                    continue

                # Nome e link estão na 3ª coluna (índice 2)
                celula_nome = colunas[2]
                link_tag = celula_nome.find("a", href=True)
                if not link_tag:
                    continue

                nome_ep = link_tag.get_text(strip=True)
                href    = link_tag["href"]
                url_ep  = BASE_URL + href if href.startswith("/") else href

                if nome_ep and url_ep:
                    episodios.append({
                        "temporada": temporada_atual,
                        "numero_ep": num_ep,
                        "nome":      nome_ep,
                        "url":       url_ep,
                    })

    print(f"  → {len(episodios)} episódios encontrados\n")
    return episodios

# ---------------------------------------------------------------------------
# Scraping de um episódio
# ---------------------------------------------------------------------------

def scrape_episodio(page: Page, ep_meta: dict) -> dict:
    """Visita a página do episódio e extrai infobox + todas as seções."""
    url = ep_meta["url"]
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector(".mw-parser-output", timeout=15000)
    except Exception:
        pass

    soup    = BeautifulSoup(page.content(), "html.parser")
    content = soup.find("div", class_="mw-parser-output")

    resultado = {
        **ep_meta,
        "infobox": {},
        "intro":   "",
        "secoes":  {},
    }

    if not content:
        return resultado

    resultado["infobox"] = extrair_infobox_episodio(soup)
    resultado["intro"]   = extrair_intro(content)
    resultado["secoes"]  = extrair_todas_secoes(content)
    return resultado

# ---------------------------------------------------------------------------
# Geração do Markdown RAG-ready
# ---------------------------------------------------------------------------

def gerar_markdown(ep: dict) -> str:
    nome      = ep["nome"]
    temporada = ep["temporada"]
    numero_ep = ep["numero_ep"]
    infobox   = ep.get("infobox", {})
    intro     = ep.get("intro", "")
    secoes    = ep.get("secoes", {})

    linhas = ["---"]
    linhas.append('tipo: "episodio"')
    linhas.append(f"temporada: {temporada}")
    linhas.append(f"numero_ep: {numero_ep}")
    linhas.append(f'nome: "{nome}"')

    for campo in ["nome_original", "data_exibicao", "data_exibicao_br",
                  "diretor", "roteiro", "storyboard", "audiencia", "codigo_producao"]:
        val = infobox.get(campo, "")
        if val:
            linhas.append(f'{campo}: "{val}"')

    linhas.append(f'url: "{ep["url"]}"')
    linhas.append("---")
    linhas.append("")

    # Cabeçalho rico com contexto para o embedding
    linhas.append(f"# {nome}")
    if infobox.get("nome_original"):
        linhas.append(f"_{infobox['nome_original']}_")
    linhas.append("")
    linhas.append(f"**Temporada {temporada} — Episódio {numero_ep}**")
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


def nome_arquivo(ep: dict) -> str:
    """Gera o nome do arquivo .md para um episódio."""
    return f"T{ep['temporada']:02d}_E{ep['numero_ep']:03d}_{slugify(ep['nome'])}.md"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    todos = []
    erros = []

    print(f"\n{'='*60}")
    print(f"Scraping episódios para RAG")
    print(f"Saída: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page()

        # 1. Descobre todos os episódios via Lista_de_Episódios
        episodios = scrape_lista_episodios(page)
        total = len(episodios)

        # 2. Scrape individual de cada episódio
        for i, ep_meta in enumerate(episodios, 1):
            label = f"T{ep_meta['temporada']:02d}E{ep_meta['numero_ep']:03d} — {ep_meta['nome']}"
            print(f"[{i:03d}/{total}] {label}")

            try:
                dados = scrape_episodio(page, ep_meta)
                todos.append(dados)

                md      = gerar_markdown(dados)
                arquivo = os.path.join(OUTPUT_DIR, nome_arquivo(dados))
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
                erros.append(ep_meta["nome"])
                todos.append({**ep_meta, "erro": str(e)})

            time.sleep(0.5)

        browser.close()

    # 3. JSON consolidado
    json_path = os.path.join(OUTPUT_DIR, "episodios_rag.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✔ Processados : {len(todos) - len(erros)}/{total}")
    print(f"✗ Erros       : {len(erros)} {erros if erros else ''}")
    print(f"JSON          : {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
