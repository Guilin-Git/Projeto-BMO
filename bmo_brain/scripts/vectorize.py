"""
vectorize.py
------------
Vetoriza os documentos Gold do BMO Brain para ChromaDB.

Estratégia:
  - Chunking por seção (## headers)
  - Sub-seções (### headers) viram chunks separados
  - Documentos curtos → chunk único
  - Seções longas (>2000 chars) → split por parágrafos com overlap
  - Metadata do frontmatter preservado em cada chunk
  - Embedding: intfloat/multilingual-e5-small

Uso:
  uv run vectorize.py          # vetoriza tudo
  uv run vectorize.py --query "Quem mora na Casa da Árvore?"  # testa query
"""

import json
import os
import re
import sys
import time
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

BASE_DIR = r"C:\Users\PC\Projeto-BMO\bmo_brain"
GOLD_DIR = os.path.join(BASE_DIR, "knowledge", "gold")
CHROMA_DIR = os.path.join(BASE_DIR, "database", "chroma_db")

CATEGORIAS = ["episodes", "characters", "places", "items"]
COLLECTION_NAME = "bmo_knowledge"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Limites de chunking
MIN_DOC_SIZE = 100       # chars mínimos para considerar o doc útil
MAX_CHUNK_SIZE = 2000    # chars max antes de split por parágrafo
OVERLAP_PARAGRAPHS = 1   # parágrafos de overlap ao splitar


# ---------------------------------------------------------------------------
# Parse de Markdown com frontmatter
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

                    # Parse lista YAML inline [a, b, c]
                    if valor.startswith("[") and valor.endswith("]"):
                        items = re.findall(r'"([^"]*)"', valor)
                        frontmatter[chave] = items
                    elif valor.lower() in ("true", "false"):
                        frontmatter[chave] = valor.lower() == "true"
                    else:
                        try:
                            frontmatter[chave] = int(valor)
                        except ValueError:
                            frontmatter[chave] = valor.strip('"').strip("'")

    return frontmatter, corpo


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def extrair_contexto_rag(corpo: str) -> str:
    """Extrai a frase de contexto RAG (linha > no início do corpo)."""
    for linha in corpo.strip().split("\n"):
        linha = linha.strip()
        if linha.startswith(">"):
            return linha.lstrip("> ").strip()
    return ""


def chunkar_por_secoes(corpo: str) -> list[dict]:
    """
    Divide o corpo em chunks por seção h2/h3.
    Retorna lista de {"secao": str, "texto": str}.
    """
    chunks = []
    secao_atual = None
    subsecao_atual = None
    conteudo_atual = []

    linhas = corpo.strip().split("\n")

    for linha in linhas:
        # Detecta seção h2
        h2_match = re.match(r"^## (.+)$", linha)
        h3_match = re.match(r"^### (.+)$", linha)

        if h2_match:
            # Salva chunk anterior
            if secao_atual and conteudo_atual:
                secao_label = f"{secao_atual} > {subsecao_atual}" if subsecao_atual else secao_atual
                chunks.append({
                    "secao": secao_label,
                    "texto": "\n".join(conteudo_atual).strip(),
                })
            secao_atual = h2_match.group(1).strip()
            subsecao_atual = None
            conteudo_atual = []

        elif h3_match and secao_atual:
            # Salva chunk anterior da sub-seção
            if conteudo_atual:
                secao_label = f"{secao_atual} > {subsecao_atual}" if subsecao_atual else secao_atual
                chunks.append({
                    "secao": secao_label,
                    "texto": "\n".join(conteudo_atual).strip(),
                })
            subsecao_atual = h3_match.group(1).strip()
            conteudo_atual = []

        elif secao_atual:
            # Pula linhas de contexto RAG e headers h1
            if linha.strip().startswith(">") or linha.strip().startswith("# "):
                continue
            conteudo_atual.append(linha)

    # Último chunk
    if secao_atual and conteudo_atual:
        secao_label = f"{secao_atual} > {subsecao_atual}" if subsecao_atual else secao_atual
        chunks.append({
            "secao": secao_label,
            "texto": "\n".join(conteudo_atual).strip(),
        })

    return chunks


def split_chunk_longo(texto: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Divide texto longo em partes menores por parágrafo, com overlap."""
    if len(texto) <= max_size:
        return [texto]

    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]

    if len(paragrafos) <= 1:
        # Texto sem parágrafos, retorna inteiro
        return [texto]

    partes = []
    parte_atual = []
    tamanho_atual = 0

    for i, paragrafo in enumerate(paragrafos):
        if tamanho_atual + len(paragrafo) > max_size and parte_atual:
            partes.append("\n\n".join(parte_atual))
            # Overlap: mantém o último parágrafo
            if OVERLAP_PARAGRAPHS > 0:
                parte_atual = parte_atual[-OVERLAP_PARAGRAPHS:]
                tamanho_atual = sum(len(p) for p in parte_atual)
            else:
                parte_atual = []
                tamanho_atual = 0

        parte_atual.append(paragrafo)
        tamanho_atual += len(paragrafo)

    if parte_atual:
        partes.append("\n\n".join(parte_atual))

    return partes


def processar_documento(caminho: str, categoria: str) -> list[dict]:
    """
    Processa um documento Gold e retorna lista de chunks prontos para indexação.
    Cada chunk: {"id", "text", "metadata"}
    """
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()

    frontmatter, corpo = parse_md(conteudo)
    nome_arquivo = os.path.basename(caminho)
    nome_base = nome_arquivo.replace(".md", "")

    # Extrair contexto RAG
    contexto = extrair_contexto_rag(corpo)

    # Texto real (sem markdown/headers)
    texto_real = re.sub(r"[#>\n\r\s*_\-]", "", corpo)

    # Documento muito curto → chunk único (ou pular se sem conteúdo)
    if len(texto_real) < MIN_DOC_SIZE:
        # Ainda gera chunk, mas com flag incompleto
        texto_corpo = corpo.strip()
        # Remove headers e linhas de contexto
        texto_corpo = re.sub(r"^[#>].*$", "", texto_corpo, flags=re.MULTILINE).strip()
        if not texto_corpo:
            return []

        chunk_id = f"{categoria}/{nome_base}/geral"
        text_para_embedding = f"passage: {contexto} {texto_corpo}" if contexto else f"passage: {texto_corpo}"

        return [{
            "id": chunk_id,
            "text": text_para_embedding,
            "metadata": _build_metadata(frontmatter, categoria, nome_arquivo, "Geral", incompleto=True),
        }]

    # Chunkar por seções
    secoes = chunkar_por_secoes(corpo)

    if not secoes:
        # Documento sem seções → chunk único
        texto_corpo = corpo.strip()
        texto_corpo = re.sub(r"^[#>].*$", "", texto_corpo, flags=re.MULTILINE).strip()
        if not texto_corpo:
            return []

        chunk_id = f"{categoria}/{nome_base}/geral"
        text_para_embedding = f"passage: {contexto} {texto_corpo}" if contexto else f"passage: {texto_corpo}"

        return [{
            "id": chunk_id,
            "text": text_para_embedding,
            "metadata": _build_metadata(frontmatter, categoria, nome_arquivo, "Geral"),
        }]

    # Gerar chunks
    chunks = []
    for secao_info in secoes:
        secao = secao_info["secao"]
        texto = secao_info["texto"]

        if not texto.strip():
            continue

        # Limpar linhas de lista (- item) para texto corrido se necessário
        # Manter como está — listas são informativas

        # Split se muito longo
        partes = split_chunk_longo(texto)

        for i, parte in enumerate(partes):
            secao_slug = re.sub(r"[^a-zA-Z0-9_]", "_", secao.lower()).strip("_")
            if len(partes) > 1:
                chunk_id = f"{categoria}/{nome_base}/{secao_slug}_p{i+1}"
            else:
                chunk_id = f"{categoria}/{nome_base}/{secao_slug}"

            # Formato e5: prefixo "passage:" para documentos
            text_para_embedding = f"passage: {contexto} {parte}" if contexto else f"passage: {parte}"

            chunks.append({
                "id": chunk_id,
                "text": text_para_embedding,
                "metadata": _build_metadata(
                    frontmatter, categoria, nome_arquivo, secao,
                    incompleto=frontmatter.get("incompleto", False),
                ),
            })

    return chunks


def _build_metadata(
    frontmatter: dict,
    categoria: str,
    nome_arquivo: str,
    secao: str,
    incompleto: bool = False,
) -> dict:
    """Constrói metadata para um chunk."""
    meta = {
        "categoria": categoria,
        "source_file": nome_arquivo,
        "secao": secao,
    }

    # Campos do frontmatter que queremos preservar como metadata
    campos_meta = [
        "tipo", "nome", "temporada", "numero_ep", "resumo",
        "link", "categoria",  # categoria do personagem, ex: "Protagonistas"
    ]
    for campo in campos_meta:
        if campo in frontmatter:
            val = frontmatter[campo]
            if isinstance(val, (str, int, float, bool)):
                if campo == "categoria":
                    meta["subcategoria"] = str(val)
                else:
                    meta[campo] = str(val) if not isinstance(val, (int, float, bool)) else val
            elif isinstance(val, list):
                meta[campo] = ", ".join(str(v) for v in val)

    # Tags como string (ChromaDB não suporta listas em metadata)
    if "tags" in frontmatter and isinstance(frontmatter["tags"], list):
        meta["tags"] = ", ".join(frontmatter["tags"])

    # Cross-refs
    if "personagens_mencionados" in frontmatter and isinstance(frontmatter["personagens_mencionados"], list):
        meta["personagens_mencionados"] = ", ".join(frontmatter["personagens_mencionados"])

    if incompleto:
        meta["incompleto"] = True

    return meta


# ---------------------------------------------------------------------------
# Indexação no ChromaDB
# ---------------------------------------------------------------------------

def indexar_chunks(chunks: list[dict]):
    """Indexa todos os chunks no ChromaDB."""
    import chromadb

    print(f"\n📦 Inicializando ChromaDB em: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Deleta collection anterior se existir
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  ♻ Collection anterior removida")
    except Exception:
        pass

    # Cria nova collection com embedding function
    print(f"  🔧 Carregando modelo: {EMBEDDING_MODEL}")

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Indexar em batches
    BATCH_SIZE = 50
    total = len(chunks)

    print(f"\n🚀 Indexando {total} chunks...")
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        processed = min(i + BATCH_SIZE, total)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        print(f"  [{processed:4d}/{total}] {processed*100//total}% | {rate:.1f} chunks/s | ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n✅ {total} chunks indexados em {elapsed:.1f}s")
    print(f"   ChromaDB salvo em: {CHROMA_DIR}")

    return collection, embedding_fn


# ---------------------------------------------------------------------------
# Query de teste
# ---------------------------------------------------------------------------

def testar_query(query: str, n_results: int = 5):
    """Executa uma query de similaridade no ChromaDB."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Formato e5: prefixo "query:" para queries
    query_text = f"query: {query}"

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\n🔍 Query: \"{query}\"")
    print(f"{'='*60}")

    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = 1 - dist  # cosine distance → similarity
        print(f"\n  [{i+1}] Score: {similarity:.3f}")
        print(f"      ID: {doc_id}")
        print(f"      Nome: {meta.get('nome', 'N/A')} | Seção: {meta.get('secao', 'N/A')}")
        print(f"      Tags: {meta.get('tags', 'N/A')}")
        # Mostra primeiros 200 chars do texto (removendo prefixo "passage:")
        texto_preview = doc.replace("passage: ", "", 1)[:200]
        print(f"      Texto: {texto_preview}...")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vetoriza dados Gold para ChromaDB")
    parser.add_argument("--query", type=str, help="Testa uma query de similaridade")
    parser.add_argument("--n-results", type=int, default=5, help="Número de resultados na query")
    args = parser.parse_args()

    if args.query:
        testar_query(args.query, n_results=args.n_results)
        return

    # --- Pipeline de vetorização ---
    all_chunks = []
    stats = {cat: {"docs": 0, "chunks": 0, "skipped": 0} for cat in CATEGORIAS}

    for categoria in CATEGORIAS:
        cat_dir = os.path.join(GOLD_DIR, categoria)
        if not os.path.exists(cat_dir):
            continue

        arquivos = sorted(f for f in os.listdir(cat_dir) if f.endswith(".md"))
        print(f"\n{'='*60}")
        print(f"📂 {categoria.upper()} ({len(arquivos)} arquivos)")
        print(f"{'='*60}")

        for nome_arquivo in arquivos:
            caminho = os.path.join(cat_dir, nome_arquivo)
            chunks = processar_documento(caminho, categoria)

            if chunks:
                all_chunks.extend(chunks)
                stats[categoria]["docs"] += 1
                stats[categoria]["chunks"] += len(chunks)
            else:
                stats[categoria]["skipped"] += 1

    # Relatório de chunking
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO DE CHUNKING")
    print(f"{'='*60}")
    total_docs = 0
    total_chunks = 0
    total_skipped = 0
    for cat in CATEGORIAS:
        s = stats[cat]
        total_docs += s["docs"]
        total_chunks += s["chunks"]
        total_skipped += s["skipped"]
        skip_info = f" ({s['skipped']} ignorados)" if s["skipped"] else ""
        print(f"  {cat:15s}: {s['docs']:3d} docs → {s['chunks']:4d} chunks{skip_info}")
    print(f"  {'TOTAL':15s}: {total_docs:3d} docs → {total_chunks:4d} chunks ({total_skipped} ignorados)")

    # Indexar
    if all_chunks:
        indexar_chunks(all_chunks)

        # Teste rápido
        print("\n" + "="*60)
        print("🧪 TESTE RÁPIDO")
        testar_query("Quem mora na Casa da Árvore?")
        testar_query("Quem é o Lich?")


if __name__ == "__main__":
    main()
