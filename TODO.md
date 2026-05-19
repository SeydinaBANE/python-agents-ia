# TODO — Feuille de route Python & Agents IA

## Module 1 — Python Fondations ✅
- [x] Types & variables (int, str, list, dict, set, tuple)
- [x] Contrôle de flux (if/elif/else, for, while)
- [x] Fonctions (def, *args, **kwargs, lambda)
- [x] Gestion d'erreurs (try/except/finally, exceptions custom)
- [x] Fichiers & I/O (open, pathlib, json)
- [x] Modules & packages (import, pip, venv)
- [x] **Projet** : chatbot CLI avec historique JSON → `01_fondations/projet_chatbot.py`

## Module 2 — Python Intermédiaire ✅
- [x] POO (class, héritage, méthodes magiques __str__, __repr__, __eq__)
- [x] Décorateurs (@property, @staticmethod, custom)
- [x] Générateurs (yield, comprehensions)
- [x] Context managers (with, __enter__/__exit__)
- [x] **Async/Await** (asyncio, async def, await, asyncio.gather)
- [x] Type hints (Optional, List, Dict, Union, TypedDict)
- [x] **Projet** : client async appelant 3 modèles en parallèle → `02_intermediaire/projet_async_client.py`

## Module 3 — Écosystème IA ✅
- [x] pydantic v2 (BaseModel, validation, schémas)
- [x] httpx (requêtes HTTP async)
- [x] python-dotenv (gestion .env)
- [x] loguru (logging structuré)
- [x] rich (affichage terminal)
- [x] typer (CLI)
- [x] **Projet** : CLI de démo avec check/ask/benchmark → `03_ecosysteme_ia/projet_setup_demo.py`

## Module 4 — LLM APIs ✅
- [x] OpenRouter (proxy unifié via SDK openai)
- [x] System prompt & conversation history
- [x] Streaming responses (SSE + Rich Live)
- [x] **Tool use / Function calling** (boucle tool_calls complète)
- [x] Prompt caching (cache_control)
- [x] Multi-modèles (haiku vs sonnet)
- [x] **Projet** : agent complet avec mémoire et streaming → `04_llm_apis/projet_agent_complet.py`

## Module 5 — Architecture Agent from Scratch ✅
- [x] Tool registry (décorateur @tool auto-JSON Schema)
- [x] Mémoire conversationnelle (MemoireCourte, troncature)
- [x] Mémoire longue (MemoireLongue, JSON persistence)
- [x] Boucle ReAct (Reason + Act + Observation + Answer)
- [x] Structured output (JSON mode)
- [x] **Projet** : agent ReAct avec /memoire /trace /reset → `05_agent_architecture/projet_agent_react.py`

## Module 6 — Frameworks d'Agents ✅
- [x] LangChain — LCEL pipeline, @tool, AgentExecutor, mémoire LCEL
- [x] LangGraph — StateGraph, EtatAgent TypedDict, routing conditionnel
- [x] CrewAI — Agent/Task/Crew, BaseTool, Process.sequential/parallel
- [x] **Projet** : agent de recherche LangGraph → `06_frameworks/projet_agent_recherche.py`

## Module 7 — Production & MCP ✅
- [x] MCP (Model Context Protocol) — serveur MCP complet → `07_production/04_mcp_server.py`
- [x] RAG complet (ChromaDB, chunking, embeddings) → `07_production/03_rag.py`
- [x] FastAPI — API REST avec sessions et streaming → `07_production/01_fastapi.py`
- [x] Docker — multi-stage build avec uv → `Dockerfile`
- [x] Observabilité (loguru, métriques en mémoire) → `07_production/05_logging.py`
- [x] Tests (pytest, mock LLMs, TestClient) → `07_production/02_tests.py`
- [x] **Projet final** : API production complète → `07_production/projet_api_complete.py`

---

## Progression
Commence par le Module 1, mais dès le Module 2 terminé, **saute au Module 4** pour
toucher les LLMs rapidement — ça motive. Reviens au Module 3 en parallèle.
