# TODO — Feuille de route Python & Agents IA

## Module 1 — Python Fondations
- [ ] Types & variables (int, str, list, dict, set, tuple)
- [ ] Contrôle de flux (if/elif/else, for, while)
- [ ] Fonctions (def, *args, **kwargs, lambda)
- [ ] Gestion d'erreurs (try/except/finally, exceptions custom)
- [ ] Fichiers & I/O (open, pathlib, json)
- [ ] Modules & packages (import, pip, venv)
- [ ] **Projet** : chatbot CLI avec historique JSON

## Module 2 — Python Intermédiaire
- [ ] POO (class, héritage, méthodes magiques __str__, __repr__, __eq__)
- [ ] Décorateurs (@property, @staticmethod, custom)
- [ ] Générateurs (yield, comprehensions)
- [ ] Context managers (with, __enter__/__exit__)
- [ ] **Async/Await** (asyncio, async def, await, aiohttp)
- [ ] Type hints (Optional, List, Dict, Union, TypedDict)
- [ ] **Projet** : client async appelant 3 APIs en parallèle

## Module 3 — Écosystème IA
- [ ] pydantic v2 (BaseModel, validation, schémas)
- [ ] httpx (requêtes HTTP async)
- [ ] python-dotenv (gestion .env)
- [ ] loguru (logging structuré)
- [ ] rich (affichage terminal)
- [ ] typer (CLI)

## Module 4 — LLM APIs
- [ ] Anthropic Claude API — messages de base
- [ ] System prompt & conversation history
- [ ] Streaming responses
- [ ] **Tool use / Function calling**
- [ ] Prompt caching (cache_control)
- [ ] Batch API
- [ ] OpenAI API (complémentaire)

## Module 5 — Architecture Agent from Scratch
- [ ] Tool registry (décorateur @tool)
- [ ] Mémoire conversationnelle (court terme)
- [ ] Mémoire longue (vector store)
- [ ] Boucle ReAct (Reason + Act)
- [ ] Structured output (JSON mode)
- [ ] **Projet** : agent météo avec tools réels

## Module 6 — Frameworks d'Agents
- [ ] LangChain — agent avec RAG
- [ ] LlamaIndex — indexation de documents
- [ ] CrewAI — multi-agents avec rôles
- [ ] AutoGen — agents conversationnels
- [ ] **Projet** : agent de recherche (web → résumé → markdown)

## Module 7 — Production & MCP
- [ ] MCP (Model Context Protocol) — créer un serveur MCP
- [ ] RAG complet (ChromaDB ou Supabase pgvector)
- [ ] FastAPI — exposer l'agent comme API REST
- [ ] Docker — conteneuriser l'agent
- [ ] Observabilité (tracer les coûts LLM, latence)
- [ ] Tests (mocker LLMs, tester tools)
- [ ] **Projet final** : agent déployé avec endpoint /chat

---

## Progression
Commence par le Module 1, mais dès le Module 2 terminé, **saute au Module 4** pour
toucher les LLMs rapidement — ça motive. Reviens au Module 3 en parallèle.
