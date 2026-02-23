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
LIST_URL = f"{BASE_URL}/pt-br/wiki/Lista_de_Epis%C3%B3dios"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(texto: str) -> str:
    """Converte uma string em um slug seguro para nome de arquivo."""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s-]", "", texto).strip()
    texto = re.sub(r"[\s]+", "_", texto)
    return texto[:80]  # Limita o tamanho


def extrair_numero_temporada(texto_header: str) -> int | None:
    """Extrai número da temporada do texto do header h2."""
    match = re.search(r"(\d+)[ªº°]?\s*[Tt]emporada", texto_header)
    if match:
        return int(match.group(1))
    if "Piloto" in texto_header:
        return 0
    return None


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


def lista_apos_header(soup_content, nome_secao: str, nivel="h3") -> list[str]:
    """
    Extrai os itens <li> de uma <ul> logo após um header com o texto dado.
    """
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


# ---------------------------------------------------------------------------
# Scraping das páginas
# ---------------------------------------------------------------------------

def scrape_lista_episodios(page: Page) -> list[dict]:
    """Extrai a lista de todos os episódios da página principal."""
    print(f"Buscando lista de episódios: {LIST_URL}")
    soup = navegar(page, LIST_URL)

    content = soup.find("div", class_="mw-parser-output")
    if not content:
        raise RuntimeError("Não foi possível encontrar o conteúdo principal da lista.")

    episodios = []
    temporada_atual = None

    for elemento in content.children:
        if elemento.name == "h2":
            span = elemento.find("span", class_="mw-headline")
            if span:
                num = extrair_numero_temporada(span.get_text(strip=True))
                if num is not None:
                    temporada_atual = num
                    print(f"  -> Temporada {temporada_atual}: {span.get_text(strip=True)}")

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
                num_ep_texto = colunas[0].get_text(strip=True)
                try:
                    num_ep = int(num_ep_texto)
                except ValueError:
                    continue  # Ignora linhas que não têm número de ep inteiro

                celula_nome = colunas[2]
                link_tag = celula_nome.find("a", href=True)
                if not link_tag:
                    continue

                nome_ep = link_tag.get_text(strip=True)
                href = link_tag["href"]
                link_ep = BASE_URL + href if href.startswith("/") else href

                if nome_ep and link_ep:
                    episodios.append({
                        "temporada": temporada_atual,
                        "numero_ep": num_ep,
                        "nome": nome_ep,
                        "link": link_ep,
                    })

    return episodios


def scrape_episodio(page: Page, ep_meta: dict) -> dict:
    """
    Visita a página do episódio e extrai todos os detalhes.
    Retorna um dict com os dados enriquecidos.
    """
    url = ep_meta["link"]
    soup = navegar(page, url)
    content = soup.find("div", class_="mw-parser-output")

    if not content:
        return ep_meta  # Retorna só o que já tínhamos

    dados = dict(ep_meta)  # Copia os metadados da lista

    # --- Infobox ---
    infobox = soup.find("aside", class_="portable-infobox")
    if infobox:
        for item in infobox.find_all("div", class_="pi-item"):
            label_tag = item.find("h3", class_="pi-data-label")
            value_tag = item.find("div", class_="pi-data-value")
            if label_tag and value_tag:
                label = label_tag.get_text(strip=True).lower()
                value = value_tag.get_text(separator=" ", strip=True)
                # Mapeia os campos comuns
                if "nome original" in label or "original" in label:
                    dados["nome_original"] = value
                elif "produção" in label or "código" in label:
                    dados["codigo_producao"] = value
                elif "exibição" in label and "brasil" in label:
                    dados["data_exibicao_br"] = value
                elif "exibição" in label:
                    dados["data_exibicao"] = value
                elif "audiência" in label:
                    dados["audiencia"] = value
                elif "diretor" in label:
                    dados["diretor"] = value
                elif "história" in label or "roteiro" in label:
                    dados["roteiro"] = value
                elif "storyboard" in label:
                    dados["storyboard"] = value

    # --- Sinopse ---
    dados["sinopse"] = texto_apos_header(content, "Sinopse", nivel="h2")

    # --- Enredo ---
    dados["enredo"] = texto_apos_header(content, "Enredo", nivel="h2")

    # --- Personagens ---
    dados["personagens_principais"] = lista_apos_header(content, "Principais", nivel="h3")
    dados["personagens_secundarios"] = lista_apos_header(content, "Secundários", nivel="h3")

    # --- Curiosidades (texto livre) ---
    dados["curiosidades"] = texto_apos_header(content, "Curiosidades", nivel="h2")

    return dados


# ---------------------------------------------------------------------------
# Geração dos arquivos .md para RAG
# ---------------------------------------------------------------------------

def gerar_md(ep: dict) -> str:
    """Gera o conteúdo Markdown RAG-ready para um episódio."""
    linhas = []

    # --- Frontmatter YAML ---
    linhas.append("---")
    linhas.append(f"temporada: {ep['temporada']}")
    linhas.append(f"numero_ep: {ep['numero_ep']}")
    linhas.append(f"nome: \"{ep['nome']}\"")
    if ep.get("nome_original"):
        linhas.append(f"nome_original: \"{ep['nome_original']}\"")
    linhas.append(f"link: \"{ep['link']}\"")
    if ep.get("data_exibicao"):
        linhas.append(f"data_exibicao: \"{ep['data_exibicao']}\"")
    if ep.get("data_exibicao_br"):
        linhas.append(f"data_exibicao_br: \"{ep['data_exibicao_br']}\"")
    if ep.get("diretor"):
        linhas.append(f"diretor: \"{ep['diretor']}\"")
    if ep.get("roteiro"):
        linhas.append(f"roteiro: \"{ep['roteiro']}\"")
    if ep.get("storyboard"):
        linhas.append(f"storyboard: \"{ep['storyboard']}\"")
    if ep.get("audiencia"):
        linhas.append(f"audiencia: \"{ep['audiencia']}\"")
    if ep.get("codigo_producao"):
        linhas.append(f"codigo_producao: \"{ep['codigo_producao']}\"")
    linhas.append("---")
    linhas.append("")

    # --- Título ---
    linhas.append(f"# {ep['nome']}")
    if ep.get("nome_original"):
        linhas.append(f"_{ep['nome_original']}_")
    linhas.append("")
    linhas.append(f"**Temporada {ep['temporada']} — Episódio {ep['numero_ep']}**")
    linhas.append("")

    # --- Sinopse ---
    if ep.get("sinopse"):
        linhas.append("## Sinopse")
        linhas.append(ep["sinopse"])
        linhas.append("")

    # --- Enredo ---
    if ep.get("enredo"):
        linhas.append("## Enredo")
        linhas.append(ep["enredo"])
        linhas.append("")

    # --- Personagens ---
    principais = ep.get("personagens_principais", [])
    secundarios = ep.get("personagens_secundarios", [])
    if principais or secundarios:
        linhas.append("## Personagens")
        if principais:
            linhas.append("### Principais")
            for p in principais:
                linhas.append(f"- {p}")
            linhas.append("")
        if secundarios:
            linhas.append("### Secundários")
            for p in secundarios:
                linhas.append(f"- {p}")
            linhas.append("")

    # --- Curiosidades ---
    if ep.get("curiosidades"):
        linhas.append("## Curiosidades")
        linhas.append(ep["curiosidades"])
        linhas.append("")

    return "\n".join(linhas)


def nome_arquivo(ep: dict) -> str:
    """Gera o nome do arquivo .md para um episódio."""
    slug = slugify(ep["nome"])
    return f"T{ep['temporada']:02d}_E{ep['numero_ep']:03d}_{slug}.md"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = "data"
    episodios_dir = os.path.join(output_dir, "episodios")
    os.makedirs(episodios_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Extrai lista de episódios
        episodios = scrape_lista_episodios(page)
        print(f"\nTotal de episódios na lista: {len(episodios)}")

        # 2. Percorre cada episódio e salva o arquivo .md
        todos_enriquecidos = []
        total = len(episodios)

        for i, ep_meta in enumerate(episodios, start=1):
            print(f"[{i:03d}/{total}] T{ep_meta['temporada']:02d}E{ep_meta['numero_ep']:03d} — {ep_meta['nome']}")

            try:
                ep = scrape_episodio(page, ep_meta)
                todos_enriquecidos.append(ep)

                # Gera e salva o arquivo .md
                conteudo_md = gerar_md(ep)
                caminho = os.path.join(episodios_dir, nome_arquivo(ep))
                with open(caminho, "w", encoding="utf-8") as f:
                    f.write(conteudo_md)

            except Exception as e:
                print(f"  ERRO: {e}")
                todos_enriquecidos.append(ep_meta)  # Salva o que tínhamos

            # Pequena pausa entre requisições
            time.sleep(0.5)

        browser.close()

    # 3. Salva JSON com todos os metadados enriquecidos
    json_path = os.path.join(output_dir, "episodios_hora_de_aventura.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos_enriquecidos, f, ensure_ascii=False, indent=2)

    arquivos_md = len([f for f in os.listdir(episodios_dir) if f.endswith(".md")])
    print(f"\n{'='*60}")
    print(f"Episódios processados : {len(todos_enriquecidos)}")
    print(f"Arquivos .md gerados  : {arquivos_md}")
    print(f"Pasta                 : {episodios_dir}")
    print(f"JSON enriquecido      : {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
