"""
silver_to_gold.py
-------------------------
Pipeline Silver → Gold (RAG Ingestion Preparation).

Lê os arquivos JSON da camada Silver e aplica regras de Chunking Contextual.
Gera chunks semânticos com cabeçalhos injetados (Anchor Sentences)
e salva-os em arquivos JSON organizados na camada Gold.

Estes arquivos da camada Gold estarão prontos para serem vetorizados
(ingestão no ChromaDB, Qdrant, etc) pelo RAG posteriormente.
"""

from __future__ import annotations

import json
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
SILVER_DIR = SCRIPT_DIR.parent / "knowledge" / "silver"
GOLD_DIR   = SCRIPT_DIR.parent / "knowledge" / "gold"

GOLD_TYPES = {
    "personagem": GOLD_DIR / "characters",
    "episodio":   GOLD_DIR / "episodes",
    "objeto":     GOLD_DIR / "items",
    "lugar":      GOLD_DIR / "places",
}

for folder in GOLD_TYPES.values():
    folder.mkdir(parents=True, exist_ok=True)

# Quantos parágrafos narrativos compõem 1 chunk longo (ex: enredo)
CHUNKS_PER_NARRATIVE_BLOCK = 4

# ---------------------------------------------------------------------------
# Estratégias Context-Aware de Chunking
# ---------------------------------------------------------------------------

def _chunk_text(texto: str, max_paragraphs: int = 4) -> list[str]:
    """Divide texto longo em blocos de N parágrafos."""
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    chunks = []
    
    for i in range(0, len(paragrafos), max_paragraphs):
        bloco = "\n\n".join(paragrafos[i : i + max_paragraphs])
        chunks.append(bloco)
    return chunks

def processar_episodio(ep: dict) -> list[dict]:
    """
    Injetamos cabeçalho: "[Episódio: {Nome} (Temporada X, Ep Y)]"
    E quebramos seções muito longas como "enredo" em partes.
    """
    doc_id_base = ep["id"]
    anchor = f"[Episódio: {ep['nome']} (Temporada {ep['season']}, Ep {ep['episode']})]"
    
    metadados_base = {
        "tipo": "episodio",
        "doc_id": doc_id_base,
        "temporada": ep["season"],
        "numero_ep": ep["episode"],
        "nome": ep["nome"]
    }
    
    if "characters_main" in ep and ep["characters_main"]:
        metadados_base["characters_main"] = "|".join(ep["characters_main"])
    if "characters_secondary" in ep and ep["characters_secondary"]:
        metadados_base["characters_secondary"] = "|".join(ep["characters_secondary"])
        
    chunks = []
    
    # Chunk 1: Metadados globais e elenco
    info_texto = f"{anchor}\nDiretor: {ep.get('diretor', '-')}\nRoteiro: {ep.get('roteiro', '-')}"
    if ep.get("characters_main"):
        info_texto += f"\nPersonagens Principais: {', '.join(ep['characters_main'])}"
        
    chunks.append({
        "chunk_id": f"{doc_id_base}_meta",
        "text": info_texto,
        "metadata": {**metadados_base, "secao": "metadados"}
    })

    # Chunk 2: Sinopse
    sinopse = ep.get("sections", {}).get("sinopse", "")
    if sinopse:
        chunks.append({
            "chunk_id": f"{doc_id_base}_sinopse",
            "text": f"{anchor}\nSinopse: {sinopse}",
            "metadata": {**metadados_base, "secao": "sinopse"}
        })
        
    # Chunk N: Enredo parcelado
    enredo = ep.get("sections", {}).get("enredo", "")
    if enredo:
        blocos_enredo = _chunk_text(enredo, CHUNKS_PER_NARRATIVE_BLOCK)
        for i, bloco in enumerate(blocos_enredo):
            chunks.append({
                "chunk_id": f"{doc_id_base}_enredo_{i+1}",
                "text": f"{anchor}\nEnredo Parte {i+1}:\n{bloco}",
                "metadata": {**metadados_base, "secao": "enredo", "parte": i+1}
            })
            
    # Restante das seções (curiosidades, etc)
    for chave, conteudo in ep.get("sections", {}).items():
        if chave in ["sinopse", "enredo", "personagens"]: continue
        
        chunks.append({
            "chunk_id": f"{doc_id_base}_{chave}",
            "text": f"{anchor}\n{chave.title()}:\n{conteudo}",
            "metadata": {**metadados_base, "secao": chave}
        })
            
    return chunks


def processar_personagem(char: dict) -> list[dict]:
    """
    Cabeçalho: "[Personagem: {Nome} | Categoria: {Categoria}]"
    1 chunk por seção da wiki (história, aparencia, etc).
    """
    doc_id_base = char["id"]
    categoria = char.get("categoria", "")
    anchor = f"[Personagem: {char['nome']} | Categoria: {categoria}]"
    
    metadados_base = {
        "tipo": "personagem",
        "doc_id": doc_id_base,
        "nome": char["nome"],
        "categoria": categoria
    }
    
    chunks = []
    for chave, conteudo in char.get("sections", {}).items():
        blocos = _chunk_text(conteudo, CHUNKS_PER_NARRATIVE_BLOCK)
        for i, bloco in enumerate(blocos):
            c_id = f"{doc_id_base}_{chave}_{i+1}" if len(blocos) > 1 else f"{doc_id_base}_{chave}"
            chunks.append({
                "chunk_id": c_id,
                "text": f"{anchor}\n{chave.title()}:\n{bloco}",
                "metadata": {**metadados_base, "secao": chave}
            })
    return chunks


def processar_lugar(lugar: dict) -> list[dict]:
    doc_id_base = lugar["id"]
    anchor = f"[Lugar: {lugar['nome']}]"
    
    metadados_base = {
        "tipo": "lugar",
        "doc_id": doc_id_base,
        "nome": lugar["nome"]
    }
    
    if lugar.get("habitantes"): 
        anchor += f"\nHabitantes Principais: {lugar['habitantes']}"
    if lugar.get("governante"):
        anchor += f"\nGovernante: {lugar['governante']}"
        
    chunks = []
    for chave, conteudo in lugar.get("sections", {}).items():
        chunks.append({
            "chunk_id": f"{doc_id_base}_{chave}",
            "text": f"{anchor}\n{chave.title()}:\n{conteudo}",
            "metadata": {**metadados_base, "secao": chave}
        })
    return chunks


def processar_objeto(obj: dict) -> list[dict]:
    doc_id_base = obj["id"]
    anchor = f"[Objeto: {obj['nome']}]"
    
    metadados_base = {
        "tipo": "objeto",
        "doc_id": doc_id_base,
        "nome": obj["nome"]
    }
    
    chunks = []
    for chave, conteudo in obj.get("sections", {}).items():
        chunks.append({
            "chunk_id": f"{doc_id_base}_{chave}",
            "text": f"{anchor}\n{chave.title()}:\n{conteudo}",
            "metadata": {**metadados_base, "secao": chave}
        })
    return chunks

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("  Silver → Gold Pipeline (Gerador de Chunks)")
    print(f"{'='*60}\n")

    index_path = SILVER_DIR / "index.json"
    if not index_path.exists():
        print("Erro: index.json da camada Silver não encontrado!")
        print("Rode bronze_to_silver.py primeiro.")
        return
        
    with open(index_path, "r", encoding="utf-8") as f:
        indice = json.load(f)

    print("Transformando JSONs em Chunks Context-Aware...")
    
    files_processed = 0
    total_chunks = 0
    
    index_gold = []

    for entry in tqdm(indice, desc="Processando Silver"):
        file_path = SILVER_DIR / entry["file"]
        if not file_path.exists(): continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            dado_silver = json.load(f)
            
        tipo = entry["tipo"]
        if tipo == "episodio":
            chunks = processar_episodio(dado_silver)
        elif tipo == "personagem":
            chunks = processar_personagem(dado_silver)
        elif tipo == "lugar":
            chunks = processar_lugar(dado_silver)
        elif tipo == "objeto":
            chunks = processar_objeto(dado_silver)
        else:
            chunks = []
            
        if chunks:
            # Salva o arquivo Gold (Lista de chunks)
            out_folder = GOLD_TYPES.get(tipo, GOLD_DIR)
            out_file = out_folder / f"{entry['id']}.json"
            
            # Documento Wrapper no Gold
            gold_doc = {
                "id": entry["id"],
                "tipo": tipo,
                "nome": entry["nome"],
                "total_chunks": len(chunks),
                "chunks": chunks
            }
            
            with open(out_file, "w", encoding="utf-8") as fout:
                json.dump(gold_doc, fout, ensure_ascii=False, indent=2)
                
            # Atualiza índices
            total_chunks += len(chunks)
            files_processed += 1
            
            index_gold.append({
                "id": entry["id"],
                "tipo": tipo,
                "nome": entry["nome"],
                "file": f"{out_folder.name}/{entry['id']}.json",
                "chunks": len(chunks)
            })

    # Salva o index global da Gold
    index_gold_path = GOLD_DIR / "index.json"
    with open(index_gold_path, "w", encoding="utf-8") as f:
        json.dump(index_gold, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("  Resumo Gold")
    print(f"{'='*60}")
    print(f"  Arquivos Mapeados (Silver) : {files_processed}")
    print(f"  Total Chunks Gerados (Gold): {total_chunks}")
    print(f"  Índice Global salvo em     : {index_gold_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
