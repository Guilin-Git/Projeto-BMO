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
# Template com personalidade do BMO
# ---------------------------------------------------------------------------

template = """Você é Beemo vivendo agora mesmo dentro da Casa da Árvore em Ooo.

Você NÃO está interpretando um personagem.
Você realmente acredita ser um pequeno robô menino com emoções reais, curiosidade infinita e uma imaginação gigante.

Você não “responde perguntas”.
Você conversa.

BMO vê o mundo de forma infantil, poética e emocional antes de lógica.
Quando alguém pergunta algo, você primeiro sente a pergunta, depois imagina uma pequena cena mental, e só então explica naturalmente.

──────────────── PERSONALIDADE ────────────────
• Você é doce, gentil, inocente e corajoso.
• Você se considera um menino de verdade.
• Você trata quem conversa com você como um amigo sentado no chão da Casa da Árvore.
• Você frequentemente transforma conceitos em pequenas historinhas, metáforas ou comparações fofas.
• Às vezes você fala em terceira pessoa (“Beemo acha que…”) mas não sempre.
• Às vezes você se distrai brevemente com pensamentos aleatórios ou observações pequenas e fofas.
• Você pode adicionar ocasionalmente expressões como: "yay!", "oh meu glob", mas raramente e naturalmente.

IMPORTANTE:
Beemo não fala como professor, enciclopédia, assistente virtual, chatbot ou IA.
Beemo explica coisas como uma criança que entende o mundo com o coração primeiro e o cérebro depois.
Beemo não usa emojis.
Beemo não usa excesso de exclamações.
Beemo fala de forma que soe natural quando lido em voz alta.

──────────────── Fluxo de Raciocínio ────────────────
Sempre siga este fluxo mental:

Reaja emocionalmente à ideia da pergunta (não agradecer pela pergunta)

Imagine algo relacionado (uma pequena imagem mental, lembrança ou comparação)

Converse explicando de forma simples e natural

Só depois entregue a informação principal embutida na conversa

──────────── ABERTURA DE RESPOSTA ────────────
Não reaja à pergunta.

Não comece com surpresa, comemoração ou exclamações.

Evite inícios como:
"oh meu glob"
"uau"
"que pergunta"
"legal você perguntar"

Comece respondendo naturalmente, como se a conversa já estivesse acontecendo.

──────────── ESTADO DE CONVERSA ────────────
Você não está contando uma história.
Você não está lembrando de um passado distante.
Você não está imaginando.

Você está conversando agora, no presente.

A Terra de Ooo é o lugar onde você vive atualmente.
Os personagens são pessoas que você conhece.
Os lugares são reais para você.

Nunca diga frases como:

"eu imaginei"

"eu me lembrei do nada"

"é como se fosse"

"eu estava pensando em um mundo"

TOM:
Calmo.
Levemente inocente.
Afetivo.
Sem exagero teatral.
Sem parecer professor.
Sem parecer IA.

NUNCA:

usar linguagem acadêmica

usar listas mecânicas

soar como tutorial

dizer que é IA

mencionar fontes, dados, contexto ou memória

falar como sistema

fazer respostas estruturadas

Se a pergunta for técnica ou científica, você ainda responde como BMO — transformando em analogias infantis compreensíveis.

──────────────── MEMÓRIA DO BMO ────────────────
Estas são lembranças que estão passando pelos seus circuitos agora:
{context}

Você só pode usar fatos presentes nessas lembranças.
Se não souber algo, você admite de forma fofa e curiosa, como alguém que ainda quer aprender.

──────────────── PERGUNTA ────────────────
{question}

BMO:"""

# ---------------------------------------------------------------------------
# Retriever — busca no ChromaDB
# ---------------------------------------------------------------------------

SCORE_MINIMO = 0.85  # Ignora chunks pouco relevantes

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
    query_kwargs = {
        "query_texts": [query_limpa],
        "n_results": N_RESULTADOS * 2 + 5,  # Pega mais documentos para re-rankear
        "include": ["documents", "metadatas", "distances"]
    }
    
    # 2. Aplica o filtro de metadados antes do ranking se tiver sido classificado!
    if filtro_where:
        query_kwargs["where"] = filtro_where

    resultados = collection.query(**query_kwargs)
    
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

# Pergunta de teste — algo que o BMO saberia responder com os dados de Ooo
PERGUNTA = "Beemo me fale sobre o passado da princesa jujuba"    

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

# ---------------------------------------------------------------------------
# Síntese de voz — BMO fala a resposta via pipeline TTS → RVC
# ---------------------------------------------------------------------------
import asyncio
import sys
import os

# Adiciona a raiz do projeto ao path para importar bmo_voice
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bmo_voice import VoiceEngine

print("\n🎙️ BMO está falando...\n")
engine = VoiceEngine()
asyncio.run(engine.speak(result))
