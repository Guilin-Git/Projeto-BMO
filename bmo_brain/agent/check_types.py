import chromadb
import os
import sys
import argparse
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def check(query: str, tipo_filtro: str = None):
    db_path = os.path.join(os.path.dirname(__file__), "..", "database", "chroma_db")
    
    class E5QueryEmbeddingFunction(SentenceTransformerEmbeddingFunction):
        def __call__(self, input: list[str]) -> list[list[float]]:
            queries = [f"query: {text}" for text in input]
            return super().__call__(queries)
            
    embedding_fn = E5QueryEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="bmo_knowledge", embedding_function=embedding_fn)
    
    # Prepara filtro (em minúsculo pois a base foi indexada assim)
    where_clause = {"tipo": tipo_filtro.lower()} if tipo_filtro else None
    
    print(f"\n🔍 Buscando por: '{query}'")
    if where_clause:
        print(f"🎯 Filtro WHERE aplicado: {where_clause}")
        res = collection.query(
            query_texts=[query],
            n_results=10,
            where=where_clause
        )
    else:
        print("🎯 Sem filtro WHERE.")
        res = collection.query(
            query_texts=[query],
            n_results=10
        )
        
    print("\nResultados:")
    if res['distances'] and res['distances'][0]:
        for i, dist in enumerate(res['distances'][0]):
            meta = res['metadatas'][0][i]
            score = 1 - dist
            print(f"[{score:.3f}] {meta.get('nome')} - {meta.get('tipo', 'Sem tipo')}")
    else:
        print("Nenhum resultado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa busca na DB")
    parser.add_argument("query", type=str, help="Pergunta para buscar")
    parser.add_argument("--tipo", type=str, default=None, help="Ex: personagem, episodio, objeto")
    args = parser.parse_args()
    
    check(args.query, args.tipo)
