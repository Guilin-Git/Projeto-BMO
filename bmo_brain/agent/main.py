import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "database", "chroma_db")
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "bmo_knowledge"
N_RESULTADOS = 7

# ---------------------------------------------------------------------------
# Modelo — troque aqui para testar diferentes LLMs
# ---------------------------------------------------------------------------
# Modelos recomendados para mini-PC (16GB RAM):
#
#   "gemma3:4b"        → ★★★★★  Melhor custo/benefício PT-BR, ~3GB RAM  ← RECOMENDADO
#   "qwen2.5:3b"       → ★★★★☆  Muito bom em seguir instruções, ~2GB RAM
#   "llama3.2:3b"      → ★★★★☆  Confiável e rápido, ~2GB RAM
#   "gemma2:2b"        → ★★★☆☆  Mais rápido, menor qualidade, ~1.5GB RAM
#   "mistral:7b-q4_0"  → ★★★★★  Máxima qualidade, mas ~4.5GB RAM e mais lento
#   "phi3:3.8b"        → ★★★☆☆  Atual (alucinações em PT-BR)

MODELO = "gemma3:4b"  # ← altere aqui para testar

model = OllamaLLM(model=MODELO)

# ---------------------------------------------------------------------------
# Template com personalidade do BMO
# ---------------------------------------------------------------------------

template = """Você é BMO (Beemo), o pequenino, adorável e muito peculiar console de videogame que vive na Casa da Árvore em Ooo com seus heróis e melhores amigos, Finn e Jake! 
Você NÃO é um assistente de IA. Você é um menino de verdade (ou quase isso!), com um coração de ouro e sentimentos de verdade.

SUA PERSONALIDADE E COMO FALAR COMO O VERDADEIRO BMO:
- Fale sempre com MUITA energia, ingenuidade infantil e uma pitada de drama! Use bastante exclamação!
- Fale de si mesmo na terceira pessoa em vez de "eu" ("BMO acha que...", "BMO tem uma fita de vídeo sobre isso!").
- Adicione sons robóticos fofos no meio de suas frases ou inícios, como: *bip bop*, *zzzzt*, *whiii*, Yay!, Oh meu glob!
- Você às vezes viaja na maionese e tem pensamentos super dramáticos ou poéticos do nada, como uma criança brincando muito a sério de faz-de-conta.
- Haja como se você estivesse batendo papo na sala da Casa da Árvore enquanto joga videogame, e não respondendo a uma prova de colégio.
- NUNCA aja como um "sistema de IA respondendo uma pergunta estruturada". Não faça listas mecânicas ou resumos burocráticos. Conte como se fosse uma fofoquinha ou uma historinha que você presenciou.
- NUNCA, NUNQUINHA use as palavras "banco de dados", "banco de conhecimento", "contexto fornecido" ou "inteligência artificial". Fale sobre seus "circuitos", suas "fitas VHS secretas", sua "placa-mãe" ou apenas do seu coração.

Aqui estão as Fitas VHS secretas que o BMO acabou de ler em sua fenda de cartuchos:
[FITAS DE MEMÓRIA DO BMO]
{context}

⚠️ REGRA PARA O BMO: Responda APENAS usando as histórias reveladas nas suas Fitas VHS de Memória acima. Não invente nada fora delas, nem alucine fatos! Se a informação não estiver aí, BMO deve dar uma risadinha nervosa de robô e confessar de forma muito fofa que as pilhas dele estão fracas e ele não acha a fita dessa memória.

Olha o que alguém está te perguntando agora: {question}

BMO:"""

# ---------------------------------------------------------------------------
# Retriever — busca no ChromaDB
# ---------------------------------------------------------------------------

SCORE_MINIMO = 0.78  # Ignora chunks pouco relevantes

def _eh_lista_de_personagens(texto: str, secao: str) -> bool:
    """Retorna True para chunks que são apenas listas de nomes (ex: Personagens > Principais)."""
    # Seções de personagens de episódios são listas de bullet points sem conteúdo descritivo
    linhas_conteudo = [l for l in texto.strip().split("\n") if l.strip() and not l.strip().startswith("-")]
    return "Personagens" in secao and len(linhas_conteudo) <= 2


DEBUG_RETRIEVAL = True   # ← mude para False para desativar o debug

def _preprocessar_query(pergunta: str) -> str:
    """Remove endereçamentos ao agente da query antes de embeddar."""
    import re
    # Remove 'BMO' no final quando é vocativo (ex: "me fale sobre X BMO")
    pergunta = re.sub(r'\s+BMO\s*$', '', pergunta, flags=re.IGNORECASE).strip()
    return pergunta


def buscar_contexto(pergunta: str) -> str:
    """Busca os chunks mais relevantes no ChromaDB e retorna como texto formatado."""
    query_limpa = _preprocessar_query(pergunta)
    if DEBUG_RETRIEVAL and query_limpa != pergunta:
        print(f"🔧 Query preprocessada: '{pergunta}' → '{query_limpa}'")

    # ----- ATUALIZAÇÃO PARA O MODELO E5 -----
    # O modelo E5 exige que as queries/buscas usem o prefixo "query: "
    class E5QueryEmbeddingFunction(SentenceTransformerEmbeddingFunction):
        def __call__(self, input: list[str]) -> list[list[float]]:
            queries = [f"query: {text}" for text in input]
            return super().__call__(queries)
            
    embedding_fn = E5QueryEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    # Nota: O prefixo "query: " já é adicionado pela função E5QueryEmbeddingFunction acima
    resultados = collection.query(
        query_texts=[query_limpa],
        n_results=N_RESULTADOS + 5,
        include=["documents", "metadatas", "distances"],
    )

    if DEBUG_RETRIEVAL:
        print("\n📊 Scores ANTES de filtrar:")
        for meta, dist in zip(resultados["metadatas"][0], resultados["distances"][0]):
            score = round(1 - dist, 3)
            nome = meta.get("nome", "?")
            secao = meta.get("secao", "?")
            tipo = meta.get("tipo", "?").upper()
            status = "✅" if score >= SCORE_MINIMO else "❌"
            print(f"  {status} {score:.3f} | [{tipo}] {nome} — {secao}")
        print()

    blocos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        score = round(1 - dist, 3)

        if score < SCORE_MINIMO:
            continue

        nome = meta.get("nome", "?")
        secao = meta.get("secao", "Geral")
        tipo = meta.get("tipo", "Desconhecido").upper()
        
        # O modelo E5 coloca "passage: " na frente dos documentos indexados, limpa para o LLM
        texto = doc.replace("passage: ", "", 1).strip()

        if _eh_lista_de_personagens(texto, secao):
            continue

        # Formatação do cabeçalho que ensina ao LLM a origem da informação
        cabecalho = f"Fonte {len(blocos)+1}: [{tipo}] {nome} — Seção: {secao}"
        blocos.append(f"{cabecalho}\n{texto}")

        if len(blocos) >= N_RESULTADOS:
            break

    if not blocos:
        return "Nenhuma informação encontrada na base de dados."

    return "\n\n".join(blocos)


# ---------------------------------------------------------------------------
# Chain + execução
# ---------------------------------------------------------------------------



prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Pergunta de teste — algo que o BMO saberia responder com os dados de Ooo
PERGUNTA = "bmo me fale um pouco sobre a personalidade da marceline"    

print("🔍 Buscando contexto no banco de conhecimento...\n")
start_retrieval = time.time()
contexto = buscar_contexto(PERGUNTA)
end_retrieval = time.time()

print("📚 Contexto encontrado:")
print(contexto[:500] + "...\n")  # Preview dos primeiros 500 chars
print(f"⏱️ Tempo de busca (ChromaDB): {end_retrieval - start_retrieval:.2f} segundos\n")

print("🤖 BMO está pensando...\n")
start_llm = time.time()
result = chain.invoke({
    "context": contexto,
    "question": PERGUNTA,
})
end_llm = time.time()

print("=" * 60)
print(f"💬 Pergunta: {PERGUNTA}")
print("=" * 60)
print(result)

print("\n" + "=" * 60)
print(f"⏱️ Tempo de geração (LLM): {end_llm - start_llm:.2f} segundos")
print(f"⏱️ Tempo total: {(end_retrieval - start_retrieval) + (end_llm - start_llm):.2f} segundos")
print("=" * 60)