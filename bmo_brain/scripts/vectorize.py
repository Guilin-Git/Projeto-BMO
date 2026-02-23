"""
vectorize.py
------------
Vetoriza os documentos Gold estruturados (.json) do BMO Brain para ChromaDB.

Estratégia:
  - Lê os arquivos JSON da camada Gold (gerados por silver_to_gold.py)
  - Os chunks já possuem cabeçalhos contextuais (Anchor Sentences)
  - Extrai os chunks e seus metadados ricos
  - Embedding: intfloat/multilingual-e5-large (usando prefixo 'passage:')

Uso:
  uv run bmo_brain/scripts/vectorize.py
  uv run bmo_brain/scripts/vectorize.py --query "Quem mora no Reino Doce?"
"""

import json
import os
import sys
import time
import argparse
import chromadb
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Caminhos e Configurações
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"C:\Users\PC\Projeto-BMO\bmo_brain")
GOLD_DIR = BASE_DIR / "knowledge" / "gold"
CHROMA_DIR = BASE_DIR / "database" / "chroma_db"

CATEGORIAS = ["episodes", "characters", "places", "items"]
COLLECTION_NAME = "bmo_knowledge"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# ---------------------------------------------------------------------------
# Processamento dos JSONs da Camada Gold
# ---------------------------------------------------------------------------

def carregar_chunks_gold() -> list[dict]:
    """Lê todos os JSONs da camada Gold e extrai os chunks."""
    all_chunks = []
    
    print(f"\n📂 Lendo chunks dos arquivos JSON da camada Gold...")
    for categoria in CATEGORIAS:
        cat_dir = GOLD_DIR / categoria
        if not cat_dir.exists():
            continue
            
        arquivos = list(cat_dir.glob("*.json"))
        for arquivo in arquivos:
            with open(arquivo, "r", encoding="utf-8") as f:
                dado_gold = json.load(f)
                
            # O silver_to_gold.py salva um envelope com "chunks"
            if "chunks" in dado_gold:
                all_chunks.extend(dado_gold["chunks"])
                
    return all_chunks

# ---------------------------------------------------------------------------
# Indexação no ChromaDB
# ---------------------------------------------------------------------------

def indexar_chunks(chunks: list[dict]):
    """Indexa todos os chunks no ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Inicializando ChromaDB em: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Deleta collection anterior se existir (reindexação total)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  ♻ Collection anterior removida")
    except Exception:
        pass

    print(f"  🔧 Carregando modelo: {EMBEDDING_MODEL}")
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    
    class E5EmbeddingFunction(SentenceTransformerEmbeddingFunction):
        def __call__(self, input: list[str]) -> list[list[float]]:
            # E5 requer prefixo 'passage: ' para documentos
            passages = [f"passage: {text}" for text in input]
            return super().__call__(passages)
            
    embedding_fn = E5EmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}, # E5 performa melhor com Cosine
    )

    # Preparar listas para inserção
    # Garantir que todos os valores de metadados sejam str, int, float ou bool
    def limpar_metadados(meta: dict) -> dict:
        limpo = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                limpo[k] = v
            elif isinstance(v, list):
                limpo[k] = "|".join(str(item) for item in v)
            elif v is None:
                continue
            else:
                limpo[k] = str(v)
        return limpo

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [limpar_metadados(c["metadata"]) for c in chunks]

    BATCH_SIZE = 64
    total = len(chunks)

    print(f"\n🚀 Indexando {total} chunks...")
    start_time = time.time()

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Upserting Batches"):
        end = min(i + BATCH_SIZE, total)
        
        batch_ids = ids[i:end]
        batch_texts = texts[i:end]
        batch_metas = metas[i:end]

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
        )

    elapsed = time.time() - start_time
    print(f"\n✅ {total} chunks indexados em {elapsed:.1f}s")
    print(f"   ChromaDB salvo em: {CHROMA_DIR}")

# ---------------------------------------------------------------------------
# Query de teste
# ---------------------------------------------------------------------------

def testar_query(query: str, n_results: int = 5):
    """Executa uma query de similaridade no ChromaDB."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    class E5QueryEmbeddingFunction(SentenceTransformerEmbeddingFunction):
        def __call__(self, input: list[str]) -> list[list[float]]:
            # E5 requer prefixo 'query: ' para buscas
            queries = [f"query: {text}" for text in input]
            return super().__call__(queries)
            
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embedding_fn = E5QueryEmbeddingFunction(model_name=EMBEDDING_MODEL)
        
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception as e:
        print(f"Erro ao acessar BD: {e}")
        print("Você executou a vetorização primeiro?")
        return

    results = collection.query(
        query_texts=[query],
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
        tipo = meta.get("tipo", "N/A").upper()
        nome = meta.get("nome", "N/A")
        secao = meta.get("secao", "N/A")
        print(f"      Doc: [{tipo}] {nome} | Seção: {secao}")
        
        texto_preview = doc.replace("\n", " ")[:200]
        print(f"      Texto: {texto_preview}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vetoriza chunks JSON Gold para ChromaDB")
    parser.add_argument("--query", type=str, help="Testa uma query de similaridade")
    parser.add_argument("--n-results", type=int, default=5, help="Número de resultados na query")
    args = parser.parse_args()

    if args.query:
        testar_query(args.query, n_results=args.n_results)
        return

    chunks = carregar_chunks_gold()
    
    if not chunks:
        print("Erro: Nenhum chunk encontrado na camada Gold.")
        print("Rode `silver_to_gold.py` primeiro.")
        return
        
    indexar_chunks(chunks)

if __name__ == "__main__":
    main()
