# Formation Python → Agents IA

Apprentissage progressif de Python orienté développement d'agents IA — des bases du langage jusqu'au déploiement en production.

## Structure

```
pro/
├── 01_fondations/          # Variables, fonctions, I/O → chatbot CLI
├── 02_intermediaire/       # POO, async, décorateurs → client multi-modèles
├── 03_ecosysteme_ia/       # pydantic, httpx, loguru, rich, typer
├── 04_llm_apis/            # Tool use, streaming, boucle agent
├── 05_agent_architecture/  # Agent from scratch : ReAct, mémoire, tool registry
├── 06_frameworks/          # LangChain, LangGraph, CrewAI
├── 07_production/          # FastAPI, Docker, MCP, RAG, tests
├── Dockerfile              # Déploiement conteneurisé (multi-stage avec uv)
└── config.py               # Client OpenRouter partagé par tous les modules
```

## Prérequis

- Python 3.11+
- Compte [OpenRouter](https://openrouter.ai) avec une clé API (`sk-or-...`)
- `uv` installé (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Installation

```bash
# 1. Cloner et entrer dans le projet
git clone https://github.com/SeydinaBANE/python-agents-ia.git
cd python-agents-ia

# 2. Créer l'environnement virtuel et installer les dépendances
uv venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows
uv sync

# 3. Configurer la clé API
cp .env.example .env
# Éditer .env :  OPENROUTER_API_KEY=sk-or-xxxxxxxx
```

> Ce projet utilise **OpenRouter** comme proxy unifié pour accéder à Claude (Anthropic),
> GPT-4 (OpenAI), Mistral et d'autres LLMs via une seule clé API et le SDK `openai`.

## Lancer les projets

```bash
# Module 1 — chatbot CLI
python 01_fondations/projet_chatbot.py

# Module 4 — agent avec tool use
python 04_llm_apis/projet_agent_complet.py

# Module 5 — agent ReAct
python 05_agent_architecture/projet_agent_react.py

# Module 7 — API production
uvicorn projet_api_complete:app --app-dir 07_production --reload --port 8000
# → Docs : http://localhost:8000/docs

# Module 7 — serveur MCP (pour Claude Desktop)
python 07_production/04_mcp_server.py
```

## Progression recommandée

| Étape | Module | Concept clé |
|-------|--------|-------------|
| 1 | Module 1 | Bases solides Python |
| 2 | Module 2 | Async/await indispensable pour les agents |
| 3 | Module 4 | Premiers appels LLM (motivation !) |
| 4 | Module 3 | Écosystème IA (pydantic, loguru...) |
| 5 | Module 5 | Construire un agent from scratch |
| 6 | Module 6 | Découvrir LangChain, LangGraph, CrewAI |
| 7 | Module 7 | Déployer en production |

## Suivi
Voir [TODO.md](TODO.md) pour la checklist complète (7 modules terminés).
Voir [RESSOURCES.md](RESSOURCES.md) pour les liens, exemples de code et packages.
