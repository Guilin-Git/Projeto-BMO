import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import BaseLLM

INTENT_PROMPT = """Você é um classificador especializado de intenção de busca para Ooo (Hora de Aventura).
Sua tarefa é analisar a pergunta do usuário e determinar qual entidade principal ele procura (como se estivesse buscando num catálogo).

Responda APENAS com uma das seguintes palavras, em maiúsculas, sem NENHUM texto extra (nada de ponto final, pontuação ou explicações):
PERSONAGEM
EPISODIO
OBJETO
LUGAR
GERAL

Regras estritas:
1. Se perguntar sobre uma pessoa, criatura, animal ou herói (ex: Finn, Jake, Princesa Jujuba, BMO, Marceline, Gunter, Lich), responda PERSONAGEM.
2. Se perguntar sobre um lugar, cidade, reino ou geografia (ex: Reino Doce, Ooo, Casa da Árvore, Noitosfera), responda LUGAR.
3. Se perguntar sobre um item, roupa, arma, livro mágico ou relíquia (ex: Enquirídio, Espada, Coroa), responda OBJETO.
4. Se perguntar especificamente sobre o que acontece num episódio específico, temporada ou resumos de capítulos, responda EPISODIO.
5. Se for uma pergunta muito abstrata, relacional (ex: 'Quem mora em tal lugar?' pode envolver vários) ou não se enquadrar claramente em nenhum, responda GERAL.

Pergunta: {question}
Intenção:"""

def classificar_intencao(pergunta: str, llm: BaseLLM) -> str:
    """
    Classifica a intenção do usuário usando o modelo instanciado, 
    para pré-filtrar a busca vetorial (Metadata filtering).
    """
    prompt = ChatPromptTemplate.from_template(INTENT_PROMPT)
    chain = prompt | llm
    
    # Processa via LLM
    resultado = chain.invoke({"question": pergunta})
    
    # O invoke geralmente retorna str para OllamaLLM ou AIMessage para ChatOllama
    if not isinstance(resultado, str):
        resultado = getattr(resultado, "content", str(resultado))
        
    resultado = resultado.strip().upper()
    
    # As 4 intenções que mapeam com precisão as opções 'tipo' no ChromaDB
    valid_intents = {"PERSONAGEM", "EPISODIO", "OBJETO", "LUGAR"}
    
    # Limpa possíveis palavras geradas a mais ("A intenção é PERSONAGEM.")
    encontrados = re.findall(r'[A-Z]+', resultado)
    for word in encontrados:
        if word in valid_intents:
            return word
            
    # Fallback caso não seja nenhuma (ou caso seja GERAL/EVENTO)
    return "GERAL"
    
def obter_filtro_chroma(intencao: str) -> dict | None:
    """Mapeia o resultado do classificador de intenção para sintaxe ChromaDB where."""
    if intencao in ("PERSONAGEM", "EPISODIO", "OBJETO", "LUGAR"):
        # Importante: os metadados no ChromaDB estão em minúsculo ('personagem', 'episodio', etc)
        return {"tipo": intencao.lower()}
    return None
