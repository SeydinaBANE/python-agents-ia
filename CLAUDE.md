# CLAUDE.md — Instructions pour Claude Code

## Projet
Formation Python → Développement d'agents IA.
Répertoire de travail : `/Users/baneseydina/Desktop/projet/pro`

## Stack technique
- **Python 3.12+** avec `uv` comme gestionnaire de packages
- **LLM principal** : Anthropic Claude (claude-sonnet-4-6 ou claude-opus-4-7)
- **Validation** : pydantic v2
- **HTTP async** : httpx
- **CLI** : typer
- **Logging** : loguru

## Conventions de code

### Style
- Type hints obligatoires sur toutes les fonctions
- Pas de commentaires évidents — seulement les WHY non-évidents
- Nommage : snake_case pour variables/fonctions, PascalCase pour classes
- Longueur de ligne max : 100 caractères

### Structure des agents
```python
# Toujours utiliser async/await pour les appels LLM
async def run_agent(user_input: str) -> str:
    ...

# Tools définis avec pydantic pour la validation
class ToolInput(BaseModel):
    query: str
    max_results: int = 5
```

### Clés API
- Toujours lire depuis `.env` via `python-dotenv`
- Jamais hardcoder une clé dans le code
- `.env` est dans `.gitignore`

## Commandes fréquentes

```bash
# Activer l'environnement
source .venv/bin/activate

# Installer les dépendances
uv add <package>

# Lancer un module
python -m 04_llm_apis.claude_direct

# Tests
pytest tests/ -v
```

## Modèles Claude à utiliser
- Développement / test : `claude-sonnet-4-6`
- Production / tâches complexes : `claude-opus-4-7`
- Tâches rapides / classification : `claude-haiku-4-5-20251001`

## Prompt caching
Activer systématiquement le cache sur les system prompts longs et les contextes RAG :
```python
{"type": "text", "text": long_system_prompt, "cache_control": {"type": "ephemeral"}}
```
