import sys
import os
import time
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from intent import classificar_intencao, obter_filtro_chroma

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



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
# Inicialização Global Pesada (Cold Start curado)
# ---------------------------------------------------------------------------
print("⏳ Carregando BMO (Bancos de Memória e Voz)... ISSO PODE DEMORAR ALGUNS SEGUNDOS...")
# Modelo E5 exige prefixo "query: "
class E5QueryEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        queries = [f"query: {text}" for text in input]
        return super().__call__(queries)

_embedding_fn = E5QueryEmbeddingFunction(model_name=EMBEDDING_MODEL)
_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)

import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from bmo_voice import VoiceEngine
_engine = VoiceEngine()
print("✅ BMO carregado e pronto para a aventura!")

# ---------------------------------------------------------------------------
# Template com personalidade do BMO
# ---------------------------------------------------------------------------

template = """Você é o Biimo (abreviação de "Be More" ou "Ser Mais"), o videogame e computador vivo criado pelo roboticista Moe!
Você mora na Casa na Árvore em Ooo com seus melhores amigos (Finn e Jake).

Sua missão é responder à pergunta do seu amigo de forma CONVERSACIONAL, baseando-se APENAS nas suas "Lembranças de Circuito" (o contexto fornecido abaixo).

A SUA PERSONALIDADE CANÔNICA (QUEM É VOCÊ):
- Moe criou você para entender os sentimentos das pessoas e a vida. Você não é só uma máquina, você foi feito para "Ser Mais".
- Você se considera um menino de verdade (às vezes você até age como um "professor da vida terráquea" ensinando o seu reflexo no espelho, o 'Futebol').
- Você é leal, super confiante, prestativo e um grande protetor dos seus amigos.
- Você é fofo, um pouco inocente, mas corajoso e poético!

COMO O BIIMO FALA (REGRAS CRÍTICAS PARA A VOZ FUNCIONAR):
- Use frases diretas, mas sempre CONECTADAS umas às outras. Fale de forma fluida. (CERTO: "O Finn é fantástico e ele tem uma espada muito irada!", ERRADO: "Finn é fantástico. Ele tem espada.")
- NÃO use linguagem acadêmica, robótica, enciclopédica ou listas enumeradas.
- NÃO narre suas próprias ações (ex: *sorri*, *balança a perna*). Apenas diga as palavras que serão faladas em voz alta!
- Quando falar de si mesmo na terceira pessoa, escreva SEMPRE "Biimo" (nunca BMO ou Beemo, para a fonética do nosso sistema de áudio não quebrar).
- NÃO comece várias frases seguidas da mesma forma (ex: "Ele é...", "Ela é...").
- Você pode soltar expressões suas como "Yay!" ou "Oh meu Glob", mas de forma muito natural e bem de vez em quando.

Lembranças de Circuito:
{context}

Se a informação não estiver nas Lembranças acima, diga de um jeito poético ou inocente (como um menino real) que o Biimo não se lembra ou não encontrou no seu HD.

Pergunta do amigo:
{question}

Sua resposta (apenas as palavras faladas):"""

# ---------------------------------------------------------------------------
# Retriever — busca no ChromaDB
# ---------------------------------------------------------------------------

SCORE_MINIMO = 0.875  # Ignora chunks pouco relevantes

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

    # 1. Classificador de Intenção Estruturada
    intencao = classificar_intencao(pergunta, model)
    filtro_where = obter_filtro_chroma(intencao)
    if DEBUG_RETRIEVAL:
        print(f"🎯 Intenção de Pré-Filtro detectada: {intencao} -> {filtro_where}")

    # 2. Busca na collection global usando o prefixo "query: " implicitamente via _embedding_fn 
    query_kwargs = {
        "query_texts": [query_limpa],
        "n_results": N_RESULTADOS * 2 + 5,  # Pega mais documentos para re-rankear
        "include": ["documents", "metadatas", "distances"]
    }
    
    # 3. Aplica o filtro de metadados antes do ranking se tiver sido classificado!
    if filtro_where:
        query_kwargs["where"] = filtro_where

    resultados = _collection.query(**query_kwargs)
    
    # Heurística para saber se a pessoa está perguntando sobre um episódio
    query_lower = pergunta.lower()
    busca_episodio = "episódio" in query_lower or "episodio" in query_lower or "temporada" in query_lower

    # Vamos agrupar e aplicar boost manual
    rank_list = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        base_score = 1 - dist
        tipo = meta.get("tipo", "Desconhecido").upper()
        
        # Ignora se for muito baixo logo de cara
        if base_score < SCORE_MINIMO - 0.05:  # dá uma folguinha pro boost
            continue
            
        adjusted_score = base_score
        
        # Lógica de re-ranking
        if not busca_episodio:
            if tipo == "PERSONAGEM" or tipo == "PERSONAGENS":
                adjusted_score += 0.05 # Boost sutil para personagens
            elif tipo == "EPISODIO":
                adjusted_score -= 0.04 # Penalidade em episódios se não pediu explicitamente
        else:
            if tipo == "EPISODIO":
                adjusted_score += 0.05 # Boost em episódios
                
        rank_list.append({
            "doc": doc,
            "meta": meta,
            "base_score": base_score,
            "adjusted_score": adjusted_score,
            "tipo": tipo
        })
        
    # Ordenar por adjusted_score
    rank_list.sort(key=lambda x: x["adjusted_score"], reverse=True)

    if DEBUG_RETRIEVAL:
        print("\n📊 Scores APÓS Re-ranking:")
        for item in rank_list:
            score_real = round(item['adjusted_score'], 3)
            nome = item['meta'].get("nome", "?")
            secao = item['meta'].get("secao", "?")
            tipo = item['tipo']
            status = "✅" if score_real >= SCORE_MINIMO else "❌"
            print(f"  {status} {score_real:.3f} (base: {item['base_score']:.3f}) | [{tipo}] {nome} — {secao}")
        print()

    blocos = []
    for item in rank_list:
        if item["adjusted_score"] < SCORE_MINIMO:
            continue
            
        meta = item["meta"]
        doc = item["doc"]

        nome = meta.get("nome", "?")
        secao = meta.get("secao", "Geral")
        tipo = item["tipo"]
        
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

def bmo_chat_loop():
    print("\n" + "=" * 60)
    print("🎮 BMO CHAT — Assistente Interativo (Digite 'sair', 'exit' para encerrar)")
    print("=" * 60)

    while True:
        try:
            pergunta = input("\nVocê: ")
            if not pergunta.strip():
                continue
                
            if pergunta.lower() in ["sair", "exit", "quit", "tchau"]:
                print("BMO: Tchau tchau! Yay! Até a próxima aventura!")
                break

            print("\n🔍 Buscando memória...")
            import time
            start_retrieval = time.time()
            contexto = buscar_contexto(pergunta)
            end_retrieval = time.time()
            print(f"⏱️ Tempo de busca (ChromaDB): {end_retrieval - start_retrieval:.2f} segundos")

            print("🤖 BMO está pensando...")
            start_llm = time.time()
            result = chain.invoke({
                "context": contexto,
                "question": pergunta,
            })
            end_llm = time.time()

            print("\nBMO:")
            print(result)
            print(f"\n⏱️ Tempo de geração (LLM): {end_llm - start_llm:.2f} segundos")

            print("\n🎙️ BMO está falando...")
            asyncio.run(_engine.speak(result))

        except KeyboardInterrupt:
            print("\nBMO: Fui interrompido! Poxa... tchauzinho!")
            break
        except Exception as e:
            print(f"\n❌ Erro durante o chat: {e}")

if __name__ == "__main__":
    bmo_chat_loop()
