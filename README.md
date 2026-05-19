# Formation Python → Agents IA

Apprentissage progressif de Python orienté développement d'agents IA.

## Structure
```
pro/
├── 01_fondations/          # Variables, fonctions, I/O
├── 02_intermediaire/       # POO, async, décorateurs
├── 03_ecosysteme_ia/       # pydantic, httpx, loguru
├── 04_llm_apis/            # Claude API, tool use, streaming
├── 05_agent_architecture/  # Agent from scratch (ReAct loop)
├── 06_frameworks/          # LangChain, CrewAI, AutoGen
└── 07_production/          # FastAPI, Docker, MCP, RAG
```

## Installation

```bash
# 1. Installer uv (gestionnaire de packages)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Créer l'environnement virtuel
uv venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# 3. Installer les dépendances de base
uv add anthropic pydantic python-dotenv httpx loguru rich typer

# 4. Configurer les clés API
cp .env.example .env
# Éditer .env avec tes vraies clés
```

## Progression recommandée

1. **Module 1** → Bases solides
2. **Module 2** → Async indispensable pour les agents
3. **Module 4** → Premiers appels LLM (motivation !)
4. **Module 3** → Outils en parallèle
5. **Module 5** → Construire un vrai agent
6. **Module 6** → Découvrir les frameworks
7. **Module 7** → Mettre en production

## Suivi
Voir [TODO.md](TODO.md) pour la checklist détaillée.
Voir [RESSOURCES.md](RESSOURCES.md) pour les liens et exemples de code.
