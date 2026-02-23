# 🤖 BMO Brain — Base de Conhecimento para Agente IA

> *"Eu sou o BMO! Eu sei tudo sobre Ooo!"*

BMO Brain é a base de conhecimento do agente IA BMO — um assistente com personalidade do personagem BMO da série **Hora de Aventura (Adventure Time)**. O projeto implementa um pipeline completo de coleta, limpeza, enriquecimento e vetorização de dados para alimentar um sistema **RAG (Retrieval-Augmented Generation)**.

---

## 📐 Arquitetura

```
Fandom Wiki (PT-BR)
       │
       ▼
  [Scrapers] ──────────────────► knowledge/bronze/   (dados brutos)
       │
  [process_silver.py] ─────────► knowledge/silver/   (limpos e normalizados)
       │
  [process_gold.py] ───────────► knowledge/gold/     (enriquecidos para RAG)
       │
  [vectorize.py] ──────────────► database/chroma_db/ (embeddings indexados)
       │
       ▼
  Agente BMO (RAG + LLM)
```

---

## 🗂️ Estrutura do Projeto

```
Projeto-BMO/
├── bmo_brain/
│   ├── knowledge/
│   │   ├── bronze/          # Dados brutos dos scrapers
│   │   │   ├── episodes/    # 293 episódios
│   │   │   ├── characters/  # 73 personagens
│   │   │   ├── places/      # 116 lugares
│   │   │   └── items/       # 133 objetos
│   │   ├── silver/          # Dados limpos
│   │   └── gold/            # Dados enriquecidos (use estes no RAG)
│   ├── database/
│   │   └── chroma_db/       # ChromaDB com 1842 chunks vetorizados
│   └── scripts/
│       ├── scrape_episodes.py
│       ├── scrape_characters_ptbr.py
│       ├── scrape_places.py
│       ├── scrape_objects.py
│       ├── process_silver.py
│       ├── process_gold.py
│       └── vectorize.py
├── pyproject.toml
└── main.py
```

---

## ⚙️ Pipeline de Dados

### 1. Scraping — Camada Bronze

Quatro scrapers usando **Playwright** coletam dados da [Wiki de Hora de Aventura (PT-BR)](https://horadeaventura.fandom.com/pt-br):

| Script | O que coleta | Saída |
|---|---|---|
| `scrape_episodes.py` | 293 episódios (sinopse, enredo, personagens, curiosidades) | `.md` + `.json` |
| `scrape_characters_ptbr.py` | 73 personagens principais (descrição, aparência, habilidades, história) | `.md` + `.json` |
| `scrape_places.py` | 116 lugares (descrição, aparência, história) | `.md` + `.json` |
| `scrape_objects.py` | 133 objetos (descrição, histórico) | `.md` + `.json` |

Cada arquivo gerado é um **Markdown com frontmatter YAML**, ex:
```yaml
---
tipo: "personagem"
nome: "Finn"
categoria: "Protagonistas"
link: "https://horadeaventura.fandom.com/pt-br/wiki/Finn"
---
# Finn
## Descrição
Finn Mertens é o protagonista de Hora de Aventura...
```

```bash
uv run bmo_brain/scripts/scrape_episodes.py
uv run bmo_brain/scripts/scrape_characters_ptbr.py
uv run bmo_brain/scripts/scrape_places.py
uv run bmo_brain/scripts/scrape_objects.py
```

---

### 2. Limpeza — Camada Silver

**`process_silver.py`** lê os dados Bronze e aplica transformações sistemáticas:

| Transformação | Problema resolvido |
|---|---|
| Remove referências wiki (`[1]`, `[2]`) | Ruído residual do scraping |
| Normaliza espaços antes de pontuação | `Finn .` → `Finn.` |
| Limpa valores da infobox | `"", Memória, ""` → `Memória` |
| Limpa labels embutidos | `Antigo dono:, Marceline` → `Antigo dono: Marceline` |
| Remove colchetes soltos | `[, 1, ]` → `` |
| Corrige `numero_ep` dos episódios | Extrai número correto do nome do arquivo |
| Marca episódios incompletos | Adiciona `incompleto: true` para eps < 400 bytes |
| Remove seções duplicadas | Detecta e remove seção cujo conteúdo está contido em outra |

```bash
uv run bmo_brain/scripts/process_silver.py
```

---

### 3. Enriquecimento — Camada Gold

**`process_gold.py`** adiciona metadados semânticos para otimizar o RAG:

| Campo adicionado | Como é gerado | Para que serve |
|---|---|---|
| `resumo` | Primeiro parágrafo (≤200 chars) | Preview rápido no retrieval |
| `tags` | Análise de conteúdo por categoria | Filtragem semântica |
| `personagens_mencionados` | Match contra 55 personagens conhecidos | Cross-referencing |
| Contexto RAG | Frase introdutória (`> Este documento...`) | Melhora qualidade dos embeddings |

Exemplos de tags geradas automaticamente:
- Personagem BMO → `["herói", "protagonista", "robô", "realeza", "combate"]`
- Item Hambo → `["mágico"]`
- Episódio T01E001 → `["ação", "morte", "temporada_1"]`

```bash
uv run bmo_brain/scripts/process_gold.py
```

---

### 4. Vetorização — ChromaDB

**`vectorize.py`** transforma os documentos Gold em vetores semânticos e os indexa no ChromaDB para retrieval eficiente.

**Modelo de Embedding:** `intfloat/multilingual-e5-small`
- Roda **localmente** (sem API key)
- Otimizado para **português**
- 384 dimensões, distância cosseno

**Estratégia de Chunking — por seção:**

```
# Finn.md (162 linhas)
  ├── chunk: characters/Finn/descri__o
  ├── chunk: characters/Finn/apar_ncia
  ├── chunk: characters/Finn/habilidades
  ├── chunk: characters/Finn/hist_ria
  ├── chunk: characters/Finn/relacionamentos___jake
  ├── chunk: characters/Finn/relacionamentos___marceline
  └── ...
```

Regras: seções longas (>2000 chars) são divididas com overlap de 1 parágrafo. Documentos sem seções viram um único chunk.

**Resultados:**

| Categoria | Docs | Chunks |
|---|---|---|
| Episódios | 293 | 1277 |
| Personagens | 73 | 203 |
| Lugares | 115 | 149 |
| Objetos | 124 | 213 |
| **Total** | **605** | **1842** |

**Queries de teste (similarity score):**

```
"Quem mora na Casa da Árvore?"
  [0.91] characters/Finn/descri__o
  [0.91] characters/Jake/descri__o
  [0.89] characters/BMO/descri__o
  [0.86] places/Casa_na_Arvore/descri__o

"Quem é o Lich?"
  [0.89] characters/Lich/descri__o
  [0.85] places/Covil_do_Lich/descri__o
  [0.85] characters/Lich/hist_ria
```

```bash
# (Re)vetorizar tudo
uv run bmo_brain/scripts/vectorize.py

# Testar queries
uv run bmo_brain/scripts/vectorize.py --query "O que é a Espada de Grama?"
uv run bmo_brain/scripts/vectorize.py --query "Qual a história de Marceline?" --n-results 10
```

---

## 🚀 Setup

```bash
# Clonar e instalar dependências
git clone <repo>
cd Projeto-BMO
uv sync

# Rodar o pipeline completo (se já tem os dados Gold)
uv run bmo_brain/scripts/vectorize.py
```

**Dependências principais:**
- `chromadb` — vector store local
- `sentence-transformers` — embeddings multilingual
- `playwright` — scraping dinâmico

---

## 🔜 Próximos Passos

- [ ] `retriever.py` — interface de consulta para o LLM
- [ ] Agente BMO com personalidade da série
- [ ] API para integração com frontend
