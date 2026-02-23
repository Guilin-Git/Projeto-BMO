import json
import os
import re
import sys
import time
import unicodedata

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

# Força UTF-8 no stdout para evitar erros de encoding no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://horadeaventura.fandom.com"
CATEGORY_URL = f"{BASE_URL}/pt-br/wiki/Categoria:Objetos"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(texto: str) -> str:
    """Converte uma string em um slug seguro para nome de arquivo."""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s-]", "", texto).strip()
    texto = re.sub(r"[\s]+", "_", texto)
    return texto[:80]


def navegar(page: Page, url: str) -> BeautifulSoup:
    """Navega para a URL e retorna o BeautifulSoup do conteúdo."""
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector(".mw-parser-output", timeout=20000)
    except Exception:
        pass
    return BeautifulSoup(page.content(), "html.parser")


def texto_apos_header(soup_content, nome_secao: str, nivel="h2") -> str:
    """
    Extrai o texto dos parágrafos <p> que vêm logo após um header com o texto dado.
    Para quando encontra o próximo header do mesmo nível.
    """
    header = None
    for tag in soup_content.find_all(nivel):
        span = tag.find("span", class_="mw-headline")
        if span and nome_secao.lower() in span.get_text(strip=True).lower():
            header = tag
            break

    if not header:
        return ""

    paragrafos = []
    for sibling in header.next_siblings:
        if sibling.name in ("h2", "h3", "h4") and nivel == "h2":
            break
        if sibling.name == "h3" and nivel == "h3":
            break
        if sibling.name == "p":
            txt = sibling.get_text(separator=" ", strip=True)
            if txt:
                paragrafos.append(txt)
    return "\n\n".join(paragrafos)


def lista_apos_header(soup_content, nome_secao: str, nivel="h2") -> list[str]:
    """Extrai os itens <li> de uma <ul> logo após um header."""
    header = None
    for tag in soup_content.find_all(nivel):
        span = tag.find("span", class_="mw-headline")
        if span and nome_secao.lower() in span.get_text(strip=True).lower():
            header = tag
            break

    if not header:
        return []

    itens = []
    for sibling in header.next_siblings:
        if sibling.name in ("h2", "h3"):
            break
        if sibling.name == "ul":
            for li in sibling.find_all("li", recursive=False):
                txt = li.get_text(separator=" ", strip=True)
                if txt:
                    itens.append(txt)
            break
    return itens


def extrair_intro(content) -> str:
    """Extrai os parágrafos introdutórios (antes do primeiro <h2>)."""
    paragrafos = []
    for child in content.children:
        if child.name == "h2":
            break
        if child.name == "p":
            txt = child.get_text(separator=" ", strip=True)
            if txt:
                paragrafos.append(txt)
    return "\n\n".join(paragrafos)


# ---------------------------------------------------------------------------
# Scraping das páginas
# ---------------------------------------------------------------------------

def scrape_links_categoria(page: Page) -> list[dict]:
    """
    Navega pela página de categoria (com paginação) e coleta todos os links
    de objetos, ignorando subcategorias.
    """
    print(f"Buscando links da categoria: {CATEGORY_URL}")
    todos_links = []
    url = CATEGORY_URL

    while url:
        print(f"  -> Lendo: {url}")
        soup = navegar(page, url)

        members = soup.find("div", class_="category-page__members")
        if not members:
            break

        for item in members.find_all("a", class_="category-page__member-link", href=True):
            href = item["href"]
            nome = item.get_text(strip=True)

            # Ignora subcategorias
            if "Categoria:" in href or "Categoria:" in nome:
                continue

            link = BASE_URL + href if href.startswith("/") else href
            todos_links.append({"nome": nome, "link": link})

        # Paginação
        next_btn = soup.find("a", class_="category-page__pagination-next")
        if next_btn and next_btn.get("href"):
            next_href = next_btn["href"]
            url = BASE_URL + next_href if next_href.startswith("/") else next_href
        else:
            url = None

    return todos_links


def scrape_objeto(page: Page, meta: dict) -> dict:
    """Visita a página do objeto e extrai todos os detalhes disponíveis."""
    url = meta["link"]
    soup = navegar(page, url)
    content = soup.find("div", class_="mw-parser-output")

    if not content:
        return meta

    dados = dict(meta)

    # --- Infobox ---
    infobox = soup.find("aside", class_="portable-infobox")
    if infobox:
        for item in infobox.find_all("div", class_="pi-item"):
            label_tag = item.find("h3", class_="pi-data-label")
            value_tag = item.find("div", class_="pi-data-value")
            if label_tag and value_tag:
                label = label_tag.get_text(strip=True).lower()
                value = value_tag.get_text(separator=", ", strip=True)

                if "uso" in label or "função" in label or "tipo" in label:
                    dados["tipo_objeto"] = value
                elif "dono" in label or "proprietário" in label or "usuário" in label:
                    dados["dono"] = value
                elif "criador" in label or "fabricante" in label:
                    dados["criador"] = value
                elif "material" in label or "feito" in label:
                    dados["material"] = value
                elif "poder" in label or "habilidade" in label:
                    dados["poderes"] = value
                elif "primeira" in label and "aparição" in label:
                    dados["primeira_aparicao"] = value
                elif "última" in label and "aparição" in label:
                    dados["ultima_aparicao"] = value
                elif "estado" in label:
                    dados["estado"] = value
                elif "localização" in label or "local" in label:
                    dados["localizacao"] = value

    # --- Introdução ---
    dados["descricao"] = extrair_intro(content)

    # --- Seções textuais ---
    dados["aparencia"] = texto_apos_header(content, "Aparência", nivel="h2")
    dados["historia"] = texto_apos_header(content, "História", nivel="h2")
    dados["poderes_texto"] = texto_apos_header(content, "Poderes", nivel="h2")
    if not dados["poderes_texto"]:
        dados["poderes_texto"] = texto_apos_header(content, "Habilidades", nivel="h2")
    dados["curiosidades"] = texto_apos_header(content, "Curiosidades", nivel="h2")

    # --- Listas ---
    dados["curiosidades_lista"] = lista_apos_header(content, "Curiosidades", nivel="h2")

    return dados


# ---------------------------------------------------------------------------
# Geração dos arquivos .md para RAG
# ---------------------------------------------------------------------------

def gerar_md(obj: dict) -> str:
    """Gera o conteúdo Markdown RAG-ready para um objeto."""
    linhas = []

    # --- Frontmatter YAML ---
    linhas.append("---")
    linhas.append('tipo: "objeto"')
    linhas.append(f'nome: "{obj["nome"]}"')
    if obj.get("tipo_objeto"):
        linhas.append(f'tipo_objeto: "{obj["tipo_objeto"]}"')
    if obj.get("dono"):
        linhas.append(f'dono: "{obj["dono"]}"')
    if obj.get("criador"):
        linhas.append(f'criador: "{obj["criador"]}"')
    if obj.get("material"):
        linhas.append(f'material: "{obj["material"]}"')
    if obj.get("poderes"):
        linhas.append(f'poderes: "{obj["poderes"]}"')
    if obj.get("estado"):
        linhas.append(f'estado: "{obj["estado"]}"')
    if obj.get("localizacao"):
        linhas.append(f'localizacao: "{obj["localizacao"]}"')
    if obj.get("primeira_aparicao"):
        linhas.append(f'primeira_aparicao: "{obj["primeira_aparicao"]}"')
    if obj.get("ultima_aparicao"):
        linhas.append(f'ultima_aparicao: "{obj["ultima_aparicao"]}"')
    linhas.append(f'link: "{obj["link"]}"')
    linhas.append("---")
    linhas.append("")

    # --- Título ---
    linhas.append(f'# {obj["nome"]}')
    linhas.append("")

    # --- Descrição (intro) ---
    if obj.get("descricao"):
        linhas.append("## Descrição")
        linhas.append(obj["descricao"])
        linhas.append("")

    # --- Aparência ---
    if obj.get("aparencia"):
        linhas.append("## Aparência")
        linhas.append(obj["aparencia"])
        linhas.append("")

    # --- História ---
    if obj.get("historia"):
        linhas.append("## História")
        linhas.append(obj["historia"])
        linhas.append("")

    # --- Poderes / Habilidades ---
    if obj.get("poderes_texto"):
        linhas.append("## Poderes e Habilidades")
        linhas.append(obj["poderes_texto"])
        linhas.append("")

    # --- Curiosidades ---
    curiosidades_texto = obj.get("curiosidades", "")
    curiosidades_lista = obj.get("curiosidades_lista", [])
    if curiosidades_texto or curiosidades_lista:
        linhas.append("## Curiosidades")
        if curiosidades_texto:
            linhas.append(curiosidades_texto)
        if curiosidades_lista:
            for c in curiosidades_lista:
                linhas.append(f"- {c}")
        linhas.append("")

    return "\n".join(linhas)


def nome_arquivo(obj: dict) -> str:
    """Gera o nome do arquivo .md para um objeto."""
    slug = slugify(obj["nome"])
    return f"{slug}.md"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = r"C:\Users\PC\Projeto-BMO\bmo_brain\knowledge\items"
    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Coleta links da categoria
        objetos_meta = scrape_links_categoria(page)
        print(f"\nTotal de objetos encontrados: {len(objetos_meta)}")

        # 2. Percorre cada objeto e salva o arquivo .md
        todos_enriquecidos = []
        total = len(objetos_meta)

        for i, meta in enumerate(objetos_meta, start=1):
            print(f"[{i:03d}/{total}] {meta['nome']}")

            try:
                objeto = scrape_objeto(page, meta)
                todos_enriquecidos.append(objeto)

                # Gera e salva o arquivo .md
                conteudo_md = gerar_md(objeto)
                caminho = os.path.join(output_dir, nome_arquivo(objeto))
                with open(caminho, "w", encoding="utf-8") as f:
                    f.write(conteudo_md)

            except Exception as e:
                print(f"  ERRO: {e}")
                todos_enriquecidos.append(meta)

            # Pequena pausa entre requisições
            time.sleep(0.5)

        browser.close()

    # 3. Salva JSON consolidado
    json_path = os.path.join(output_dir, "objetos_hora_de_aventura.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos_enriquecidos, f, ensure_ascii=False, indent=2)

    arquivos_md = len([f for f in os.listdir(output_dir) if f.endswith(".md")])
    print(f"\n{'='*60}")
    print(f"Objetos processados   : {len(todos_enriquecidos)}")
    print(f"Arquivos .md gerados  : {arquivos_md}")
    print(f"Pasta                 : {output_dir}")
    print(f"JSON enriquecido      : {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
