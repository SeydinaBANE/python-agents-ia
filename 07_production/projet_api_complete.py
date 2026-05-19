"""
MODULE 7 — Projet Final : API Agent Complète
=============================================
L'agent Atlas déployé comme API REST production-ready :
  - FastAPI avec tous les endpoints               (leçon 1)
  - Tests automatisés                             (leçon 2)
  - RAG intégré (base de connaissances)           (leçon 3)
  - Tool use + mémoire de session                 (leçon 4)
  - Logging structuré + métriques                 (leçon 5)

Lance   : uvicorn 07_production.projet_api_complete:app --reload --port 8000
Docs    : http://localhost:8000/docs
Tester  : python 07_production/projet_api_complete.py --test
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console

load_dotenv()
console = Console()

# ─── LOGGING ─────────────────────────────────────────────────────────────────

import os as _os
_os.makedirs("07_production/data", exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", colorize=True)
logger.add("07_production/data/api.log", level="DEBUG", rotation="10 MB")

# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Atlas Agent API",
    description="Agent IA complet avec RAG, tool use, et mémoire de session.",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── MÉTRIQUES (en mémoire — Prometheus en production) ───────────────────────

metriques = {"total_requetes": 0, "total_tokens": 0, "total_erreurs": 0, "duree_totale_ms": 0.0}

@app.middleware("http")
async def middleware_metriques(request: Request, call_next):
    debut = time.perf_counter()
    response = await call_next(request)
    duree_ms = (time.perf_counter() - debut) * 1000
    if request.url.path.startswith("/chat"):
        metriques["total_requetes"] += 1
        metriques["duree_totale_ms"] += duree_ms
    return response

# ─── MODÈLES PYDANTIC ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str  = Field(min_length=1, max_length=10_000)
    session_id: str  = Field(default_factory=lambda: str(uuid.uuid4()))
    use_rag:    bool = Field(default=True,  description="Utiliser la base de connaissances")
    use_tools:  bool = Field(default=True,  description="Autoriser les tool calls")
    stream:     bool = Field(default=False)

class ChatResponse(BaseModel):
    reponse:    str
    session_id: str
    tokens:     int
    duree_ms:   float
    sources_rag: list[str] = []
    tools_utilises: list[str] = []

class Metriques(BaseModel):
    total_requetes:  int
    total_tokens:    int
    total_erreurs:   int
    duree_moyenne_ms: float

# ─── OUTILS ──────────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_meteo", "description": "Météo d'une ville",
        "parameters": {"type": "object", "properties": {"ville": {"type": "string"}}, "required": ["ville"]},
    }},
    {"type": "function", "function": {
        "name": "calculer", "description": "Calcul mathématique",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "heure_actuelle", "description": "Date et heure actuelles",
        "parameters": {"type": "object", "properties": {}},
    }},
]

def executer_tool(nom: str, args: dict) -> str:
    if nom == "get_meteo":
        meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Lyon": "15°C pluvieux"}
        return meteos.get(args["ville"], "20°C inconnu")
    if nom == "calculer":
        try:
            e = args["expression"]
            if not all(c in set("0123456789+-*/()., ") for c in e):
                return "Expression non autorisée"
            return str(eval(e))  # noqa: S307
        except Exception as ex:
            return f"Erreur: {ex}"
    if nom == "heure_actuelle":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Tool '{nom}' inconnu"

# ─── RAG ──────────────────────────────────────────────────────────────────────

_kb_instance = None

def get_kb():
    global _kb_instance
    if _kb_instance is None:
        try:
            client     = chromadb.PersistentClient(path="07_production/data/chroma")
            _kb_instance = client.get_or_create_collection("agents_ia_kb")
        except Exception as e:
            logger.warning("ChromaDB indisponible : {}", e)
    return _kb_instance

def rechercher_contexte(query: str, n: int = 3) -> tuple[str, list[str]]:
    kb = get_kb()
    if kb is None or kb.count() == 0:
        return "", []
    try:
        resultats = kb.query(query_texts=[query], n_results=min(n, kb.count()))
        chunks    = resultats["documents"][0]
        titres    = [m["titre"] for m in resultats["metadatas"][0]]
        contexte  = "## Base de connaissances :\n" + "\n".join(f"- {c}" for c in chunks)
        return contexte, list(dict.fromkeys(titres))
    except Exception as e:
        logger.warning("Erreur RAG : {}", e)
        return "", []

# ─── SESSIONS ────────────────────────────────────────────────────────────────

sessions: dict[str, list[dict]] = {}

# ─── BOUCLE AGENT ────────────────────────────────────────────────────────────

async def executer_agent(req: ChatRequest) -> tuple[str, int, list[str], list[str]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY manquante")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Contexte RAG
    contexte_rag, sources = ("", [])
    if req.use_rag:
        contexte_rag, sources = rechercher_contexte(req.message)

    system = "Tu es Atlas, un assistant expert en Python et agents IA. Réponds en français."
    if contexte_rag:
        system += f"\n\n{contexte_rag}"

    historique = sessions.setdefault(req.session_id, [])
    historique.append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": system}] + historique

    tools_utilises = []
    tokens_total   = 0
    iteration      = 0

    while iteration < 6:
        iteration += 1
        kwargs = {"model": "anthropic/claude-sonnet-4-5", "messages": messages, "max_tokens": 1024}
        if req.use_tools:
            kwargs["tools"] = TOOLS_SCHEMA
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        choix    = response.choices[0]
        if response.usage:
            tokens_total += response.usage.total_tokens

        if choix.finish_reason == "stop":
            reponse = choix.message.content or ""
            historique.append({"role": "assistant", "content": reponse})
            return reponse, tokens_total, sources, tools_utilises

        if choix.finish_reason == "tool_calls":
            messages.append(choix.message)
            for tc in choix.message.tool_calls:
                nom    = tc.function.name
                args   = json.loads(tc.function.arguments)
                result = executer_tool(nom, args)
                tools_utilises.append(nom)
                logger.debug("Tool {}({}) → {}", nom, args, result[:80])
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Désolé, je n'ai pas pu répondre.", tokens_total, sources, tools_utilises

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"statut": "ok", "version": "2.0.0", "sessions": len(sessions), "kb_chunks": get_kb().count() if get_kb() else 0}

@app.get("/metriques", response_model=Metriques)
async def get_metriques():
    nb = metriques["total_requetes"]
    return Metriques(
        total_requetes=nb,
        total_tokens=metriques["total_tokens"],
        total_erreurs=metriques["total_erreurs"],
        duree_moyenne_ms=round(metriques["duree_totale_ms"] / nb, 1) if nb else 0,
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    debut = time.perf_counter()
    logger.info("Chat | session={} | message={!r}", req.session_id[:8], req.message[:60])
    try:
        reponse, tokens, sources, tools = await executer_agent(req)
        duree_ms = (time.perf_counter() - debut) * 1000
        metriques["total_tokens"] += tokens
        return ChatResponse(reponse=reponse, session_id=req.session_id, tokens=tokens, duree_ms=round(duree_ms, 1), sources_rag=sources, tools_utilises=tools)
    except HTTPException:
        metriques["total_erreurs"] += 1
        raise
    except Exception as e:
        metriques["total_erreurs"] += 1
        logger.error("Erreur chat : {}", e)
        raise HTTPException(502, f"Erreur LLM : {e}")

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    req.use_tools = False  # Le streaming ne supporte pas les tool calls ici

    async def generer() -> AsyncGenerator[str, None]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        client  = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        historique = sessions.setdefault(req.session_id, [])
        historique.append({"role": "user", "content": req.message})
        messages = [{"role": "system", "content": "Tu es Atlas, un assistant IA. Réponds en français."}] + historique
        reponse_complete = ""
        async with await client.chat.completions.create(model="anthropic/claude-haiku-4-5", messages=messages, max_tokens=1024, stream=True) as stream:
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    reponse_complete += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
        historique.append({"role": "assistant", "content": reponse_complete})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generer(), media_type="text/event-stream")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session introuvable")
    msgs = sessions[session_id]
    return {"session_id": session_id, "nb_messages": len(msgs), "messages": msgs}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session introuvable")
    del sessions[session_id]
    return {"statut": "supprimé"}

# ─── CLI POUR TESTER ─────────────────────────────────────────────────────────

async def tester_api():
    import httpx, asyncio
    BASE = "http://localhost:8000"
    async with httpx.AsyncClient() as c:
        console.print("\n[bold]Test health[/bold]")
        r = await c.get(f"{BASE}/health")
        console.print(r.json())

        console.print("\n[bold]Test chat[/bold]")
        r = await c.post(f"{BASE}/chat", json={"message": "Quel temps à Paris ?"}, timeout=30)
        console.print(r.json())

if __name__ == "__main__":
    if "--test" in sys.argv:
        import asyncio
        asyncio.run(tester_api())
    else:
        import uvicorn
        uvicorn.run("07_production.projet_api_complete:app", host="0.0.0.0", port=8000, reload=True)
