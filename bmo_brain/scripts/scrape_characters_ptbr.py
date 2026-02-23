import json
import os
import re
import sys
import time
import unicodedata
from urllib.parse import quote

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

# Força UTF-8 no stdout para evitar erros de encoding no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://horadeaventura.fandom.com"
BASE_PAGE = f"{BASE_URL}/pt-br/wiki/"

# ---------------------------------------------------------------------------
# Lista curada de personagens, agrupados por categoria
# ---------------------------------------------------------------------------

PERSONAGENS = {
    "Protagonistas": [
        ("Finn", "Finn"),
        ("Jake", "Jake"),
        ("BMO", "BMO"),
    ],
    "Núcleo principal recorrente": [
        ("Princesa Jujuba", "Princesa_Jujuba"),
        ("Marceline", "Marceline"),
        ("Lady Íris", "Lady_Íris"),
        ("Princesa Caroço", "Princesa_Caroço"),
        ("Rei Gelado", "Rei_Gelado"),
        ("Gunter", "Gunter"),
        ("Betty Grof", "Betty_Grof"),
        ("Peppermint Butler", "Peppermint_Butler"),
        ("Canelinha", "Canelinha"),
        ("Dona Tromba", "Dona_Tromba"),
        ("Sr. Porco", "Sr._Porco"),
    ],
    "Reino Doce": [
        ("Conde de Limãograb", "Conde_de_Limãograb"),
        ("Limãograb 2", "Limãograb_2"),
        ("Coronel Milho Doce", "Coronel_Milho_Doce"),
        ("Guarda Banana", "Guarda_Banana"),
        ("Chicletão", "Chicletão"),
        ("Neddy", "Neddy"),
        ("Tio Gumbald", "Tio_Gumbald"),
        ("Prima Chicle", "Prima_Chicle"),
        ("Tia Lolly", "Tia_Lolly"),
    ],
    "Reino de Fogo": [
        ("Princesa de Fogo", "Princesa_de_Fogo"),
        ("Rei do Fogo", "Rei_do_Fogo"),
        ("Flame Guardas", "Flame_Guardas"),
        ("Rapaz do Fogo", "Rapaz_do_Fogo"),
    ],
    "Vampiros / Arco Stakes": [
        ("Rei Vampiro", "Rei_Vampiro"),
        ("A Lua", "A_Lua_(Vampiros)"),
        ("A Imperatriz", "A_Imperatriz"),
        ("O Louco", "O_Louco"),
        ("A Bruxa Vampira", "Bruxa_Vampira"),
    ],
    "Humanos e Ilhas": [
        ("Minerva Campbell", "Minerva_Campbell"),
        ("Martin Mertens", "Martin_Mertens"),
        ("Susana Forte", "Susana_Forte"),
        ("Frieda", "Frieda"),
        ("Dr. Gross", "Dr._Gross"),
    ],
    "Família do Jake": [
        ("Jermaine", "Jermaine"),
        ("Charlie", "Charlie"),
        ("TV", "TV_(Personagem)"),
        ("Viola", "Viola"),
        ("Kim Kil Whan", "Kim_Kil_Whan"),
        ("Jake Jr.", "Jake_Jr."),
        ("Bronwyn", "Bronwyn"),
    ],
    "Magos e personagens mágicos": [
        ("Abracadaniel", "Abracadaniel"),
        ("Maga Caçadora", "Maga_Caçadora"),
        ("Evergreen", "Evergreen"),
        ("Paciente São Pim", "Paciente_São_Pim"),
        ("Bella Noche", "Bella_Noche"),
        ("Coconteppi", "Coconteppi"),
        ("Ash", "Ash"),
    ],
    "Vilões principais": [
        ("Lich", "Lich"),
        ("Golb", "Golb"),
        ("Orgalorg", "Orgalorg"),
        ("Ricardio", "Ricardio"),
        ("Hunson Abadeer", "Hunson_Abadeer"),
        ("Magic Man", "Magic_Man"),
    ],
    "Entidades cósmicas": [
        ("Prismo", "Prismo"),
        ("Coruja Cósmica", "Coruja_Cósmica"),
        ("Morte", "Morte"),
        ("Nova Morte", "Nova_Morte"),
        ("Grob Gob Glob Grod", "Grob_Gob_Glob_Grod"),
    ],
    "Heróis e figuras históricas de Ooo": [
        ("Billy", "Billy"),
        ("Canyon", "Canyon"),
        ("Farmworld Finn", "Farmworld_Finn"),
        ("Shoko", "Shoko"),
    ],
    "Outros recorrentes relevantes": [
        ("Caracol", "Caracol"),
        ("Braco", "Braco"),
        ("Breezinha", "Breezinha"),
        ("Cuber", "Cuber"),
        ("Shelby", "Shelby"),
        ("Tiffany Oiler", "Tiffany_Oiler"),
        ("Neptr", "Neptr"),
        ("Choose Goose", "Choose_Goose"),
    ],
}


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


def texto_completo_secao(soup_content, nome_secao: str, nivel="h2") -> str:
    """
    Extrai texto de <p> e itens de <ul>/<li> de uma seção inteira (h2),
    incluindo sub-seções h3. Útil para seções ricas como Relacionamentos.
    """
    header = None
    for tag in soup_content.find_all(nivel):
        span = tag.find("span", class_="mw-headline")
        if span and nome_secao.lower() in span.get_text(strip=True).lower():
            header = tag
            break

    if not header:
        return ""

    blocos = []
    for sibling in header.next_siblings:
        if sibling.name == "h2":
            break
        if sibling.name == "h3":
            span = sibling.find("span", class_="mw-headline")
            if span:
                blocos.append(f"\n### {span.get_text(strip=True)}")
        elif sibling.name == "p":
            txt = sibling.get_text(separator=" ", strip=True)
            if txt:
                blocos.append(txt)
        elif sibling.name == "ul":
            for li in sibling.find_all("li", recursive=False):
                txt = li.get_text(separator=" ", strip=True)
                if txt:
                    blocos.append(f"- {txt}")
    return "\n\n".join(blocos)


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


# ---------------------------------------------------------------------------
# Scraping da página de um personagem
# ---------------------------------------------------------------------------

def scrape_personagem(page: Page, nome: str, wiki_slug: str, categoria: str) -> dict:
    """Visita a página do personagem e extrai todos os detalhes."""
    url = BASE_PAGE + quote(wiki_slug, safe="/_().")
    soup = navegar(page, url)
    content = soup.find("div", class_="mw-parser-output")

    dados = {
        "nome": nome,
        "categoria": categoria,
        "link": url,
    }

    if not content:
        return dados

    # --- Infobox ---
    infobox = soup.find("aside", class_="portable-infobox")
    if infobox:
        for item in infobox.find_all("div", class_="pi-item"):
            label_tag = item.find("h3", class_="pi-data-label")
            value_tag = item.find("div", class_="pi-data-value")
            if label_tag and value_tag:
                label = label_tag.get_text(strip=True).lower()
                value = value_tag.get_text(separator=", ", strip=True)

                if "nome completo" in label or "nome" == label:
                    dados["nome_completo"] = value
                elif "apelido" in label or "alcunha" in label:
                    dados["apelidos"] = value
                elif "sexo" in label or "gênero" in label or "genero" in label:
                    dados["genero"] = value
                elif "idade" in label:
                    dados["idade"] = value
                elif "espécie" in label or "especie" in label or "raça" in label:
                    dados["especie"] = value
                elif "ocupação" in label or "ocupacao" in label:
                    dados["ocupacao"] = value
                elif "residência" in label or "residencia" in label or "lar" in label:
                    dados["residencia"] = value
                elif "parente" in label or "família" in label:
                    dados["parentes"] = value
                elif "animal" in label:
                    dados["animal_estimacao"] = value
                elif "primeira" in label and "aparição" in label:
                    dados["primeira_aparicao"] = value
                elif "última" in label and "aparição" in label:
                    dados["ultima_aparicao"] = value
                elif "voz" in label or "dublad" in label:
                    dados["dublador"] = value
                elif "estado" in label:
                    dados["estado"] = value

    # --- Introdução ---
    dados["descricao"] = extrair_intro(content)

    # --- Seções textuais ---
    dados["aparencia"] = texto_apos_header(content, "Aparência", nivel="h3")
    if not dados["aparencia"]:
        dados["aparencia"] = texto_apos_header(content, "Aparência", nivel="h2")

    dados["personalidade"] = texto_apos_header(content, "Personalidade", nivel="h2")
    if not dados["personalidade"]:
        dados["personalidade"] = texto_apos_header(content, "Personalidade", nivel="h3")

    dados["habilidades"] = texto_apos_header(content, "Habilidades", nivel="h3")
    if not dados["habilidades"]:
        dados["habilidades"] = texto_apos_header(content, "Habilidades", nivel="h2")

    dados["historia"] = texto_apos_header(content, "História", nivel="h2")

    dados["relacionamentos"] = texto_completo_secao(content, "Relacionamentos", nivel="h2")

    dados["curiosidades"] = texto_apos_header(content, "Curiosidades", nivel="h2")

    dados["citacoes"] = lista_apos_header(content, "Citações", nivel="h2")

    return dados


# ---------------------------------------------------------------------------
# Geração dos arquivos .md para RAG
# ---------------------------------------------------------------------------

def gerar_md(p: dict) -> str:
    """Gera o conteúdo Markdown RAG-ready para um personagem."""
    linhas = []

    # --- Frontmatter YAML ---
    linhas.append("---")
    linhas.append('tipo: "personagem"')
    linhas.append(f'nome: "{p["nome"]}"')
    linhas.append(f'categoria: "{p["categoria"]}"')
    if p.get("nome_completo"):
        linhas.append(f'nome_completo: "{p["nome_completo"]}"')
    if p.get("apelidos"):
        linhas.append(f'apelidos: "{p["apelidos"]}"')
    if p.get("genero"):
        linhas.append(f'genero: "{p["genero"]}"')
    if p.get("idade"):
        linhas.append(f'idade: "{p["idade"]}"')
    if p.get("especie"):
        linhas.append(f'especie: "{p["especie"]}"')
    if p.get("ocupacao"):
        linhas.append(f'ocupacao: "{p["ocupacao"]}"')
    if p.get("residencia"):
        linhas.append(f'residencia: "{p["residencia"]}"')
    if p.get("parentes"):
        linhas.append(f'parentes: "{p["parentes"]}"')
    if p.get("animal_estimacao"):
        linhas.append(f'animal_estimacao: "{p["animal_estimacao"]}"')
    if p.get("estado"):
        linhas.append(f'estado: "{p["estado"]}"')
    if p.get("primeira_aparicao"):
        linhas.append(f'primeira_aparicao: "{p["primeira_aparicao"]}"')
    if p.get("ultima_aparicao"):
        linhas.append(f'ultima_aparicao: "{p["ultima_aparicao"]}"')
    if p.get("dublador"):
        linhas.append(f'dublador: "{p["dublador"]}"')
    linhas.append(f'link: "{p["link"]}"')
    linhas.append("---")
    linhas.append("")

    # --- Título ---
    linhas.append(f'# {p["nome"]}')
    linhas.append("")

    # --- Descrição (intro) ---
    if p.get("descricao"):
        linhas.append("## Descrição")
        linhas.append(p["descricao"])
        linhas.append("")

    # --- Aparência ---
    if p.get("aparencia"):
        linhas.append("## Aparência")
        linhas.append(p["aparencia"])
        linhas.append("")

    # --- Personalidade ---
    if p.get("personalidade"):
        linhas.append("## Personalidade")
        linhas.append(p["personalidade"])
        linhas.append("")

    # --- Habilidades ---
    if p.get("habilidades"):
        linhas.append("## Habilidades")
        linhas.append(p["habilidades"])
        linhas.append("")

    # --- História ---
    if p.get("historia"):
        linhas.append("## História")
        linhas.append(p["historia"])
        linhas.append("")

    # --- Relacionamentos ---
    if p.get("relacionamentos"):
        linhas.append("## Relacionamentos")
        linhas.append(p["relacionamentos"])
        linhas.append("")

    # --- Curiosidades ---
    if p.get("curiosidades"):
        linhas.append("## Curiosidades")
        linhas.append(p["curiosidades"])
        linhas.append("")

    # --- Citações ---
    citacoes = p.get("citacoes", [])
    if citacoes:
        linhas.append("## Citações")
        for c in citacoes:
            linhas.append(f'> "{c}"')
            linhas.append("")

    return "\n".join(linhas)


def nome_arquivo(p: dict) -> str:
    """Gera o nome do arquivo .md para um personagem."""
    slug = slugify(p["nome"])
    return f"{slug}.md"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = r"C:\Users\PC\Projeto-BMO\bmo_brain\knowledge\characters"
    os.makedirs(output_dir, exist_ok=True)

    # Monta lista flat com (nome, slug, categoria)
    lista = []
    for categoria, personagens in PERSONAGENS.items():
        for nome, slug in personagens:
            lista.append((nome, slug, categoria))

    total = len(lista)
    print(f"Total de personagens para scrape: {total}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        todos_enriquecidos = []

        for i, (nome, slug, categoria) in enumerate(lista, start=1):
            print(f"[{i:03d}/{total}] {nome} ({categoria})")

            try:
                dados = scrape_personagem(page, nome, slug, categoria)
                todos_enriquecidos.append(dados)

                # Gera e salva o arquivo .md
                conteudo_md = gerar_md(dados)
                caminho = os.path.join(output_dir, nome_arquivo(dados))
                with open(caminho, "w", encoding="utf-8") as f:
                    f.write(conteudo_md)

            except Exception as e:
                print(f"  ERRO: {e}")
                todos_enriquecidos.append({
                    "nome": nome,
                    "categoria": categoria,
                    "link": BASE_PAGE + slug,
                })

            # Pequena pausa entre requisições
            time.sleep(0.5)

        browser.close()

    # Salva JSON consolidado
    json_path = os.path.join(output_dir, "personagens_hora_de_aventura.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todos_enriquecidos, f, ensure_ascii=False, indent=2)

    arquivos_md = len([f for f in os.listdir(output_dir) if f.endswith(".md")])
    print(f"\n{'='*60}")
    print(f"Personagens processados : {len(todos_enriquecidos)}")
    print(f"Arquivos .md gerados    : {arquivos_md}")
    print(f"Pasta                   : {output_dir}")
    print(f"JSON enriquecido        : {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()