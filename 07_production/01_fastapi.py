"""
MODULE 7 — Leçon 1 : FastAPI
==============================
FastAPI expose ton agent comme une API REST.
Endpoints clés : POST /chat, GET /health, GET /historique/{session_id}

Lance : uvicorn 07_production.01_fastapi:app --reload --port 8000
Docs  : http://localhost:8000/docs  (Swagger auto-généré)
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agent IA API",
    description="API REST pour interagir avec un agent IA via OpenRouter",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODÈLES PYDANTIC ────────────────────────────────────────────────────────

class MessageEntree(BaseModel):
    contenu:    str    = Field(min_length=1, max_length=10_000)
    session_id: str    = Field(default_factory=lambda: str(uuid.uuid4()))
    modele:     str    = Field(default="anthropic/claude-haiku-4-5")
    stream:     bool   = Field(default=False)

class MessageSortie(BaseModel):
    reponse:    str
    session_id: str
    modele:     str
    tokens:     int
    duree_ms:   float
    timestamp:  str

class StatutSante(BaseModel):
    statut:    str
    version:   str
    modeles:   list[str]
    api_key_ok: bool

# ─── STOCKAGE DES SESSIONS (en mémoire — Redis au Module 7 prod) ──────────────

sessions: dict[str, list[dict]] = {}

def get_historique(session_id: str) -> list[dict]:
    return sessions.setdefault(session_id, [])

def ajouter_message(session_id: str, role: str, contenu: str) -> None:
    get_historique(session_id).append({"role": role, "content": contenu})

# ─── CLIENT LLM ───────────────────────────────────────────────────────────────

def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY manquante")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

SYSTEM_PROMPT = "Tu es un assistant IA expert en Python et agents IA. Réponds en français."

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=StatutSante, tags=["Système"])
async def health():
    """Vérifie que l'API est opérationnelle."""
    return StatutSante(
        statut="ok",
        version="1.0.0",
        modeles=["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-5"],
        api_key_ok=bool(os.getenv("OPENROUTER_API_KEY")),
    )


@app.post("/chat", response_model=MessageSortie, tags=["Chat"])
async def chat(message: MessageEntree):
    """
    Envoie un message à l'agent et reçoit une réponse.

    - **contenu** : texte de la question
    - **session_id** : identifiant de session (généré automatiquement si absent)
    - **modele** : modèle LLM à utiliser
    """
    client = get_client()
    debut  = time.perf_counter()

    ajouter_message(message.session_id, "user", message.contenu)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_historique(message.session_id)

    try:
        response = await client.chat.completions.create(
            model=message.modele,
            messages=messages,
            max_tokens=1024,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur LLM : {e}")

    reponse  = response.choices[0].message.content or ""
    tokens   = response.usage.total_tokens if response.usage else 0
    duree_ms = (time.perf_counter() - debut) * 1000

    ajouter_message(message.session_id, "assistant", reponse)

    return MessageSortie(
        reponse=reponse,
        session_id=message.session_id,
        modele=message.modele,
        tokens=tokens,
        duree_ms=round(duree_ms, 1),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(message: MessageEntree):
    """Envoie un message et reçoit la réponse en streaming (Server-Sent Events)."""
    client = get_client()
    ajouter_message(message.session_id, "user", message.contenu)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_historique(message.session_id)

    async def generer() -> AsyncGenerator[str, None]:
        reponse_complete = ""
        try:
            async with await client.chat.completions.create(
                model=message.modele,
                messages=messages,
                max_tokens=1024,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        reponse_complete += token
                        # Format SSE : "data: <json>\n\n"
                        yield f"data: {json.dumps({'token': token})}\n\n"

            ajouter_message(message.session_id, "assistant", reponse_complete)
            yield f"data: {json.dumps({'done': True, 'session_id': message.session_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'erreur': str(e)})}\n\n"

    return StreamingResponse(generer(), media_type="text/event-stream")


@app.get("/historique/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Retourne l'historique d'une session."""
    historique = sessions.get(session_id)
    if historique is None:
        raise HTTPException(status_code=404, detail="Session introuvable")
    return {"session_id": session_id, "messages": historique, "nb_messages": len(historique)}


@app.delete("/historique/{session_id}", tags=["Sessions"])
async def supprimer_session(session_id: str):
    """Supprime une session et son historique."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session introuvable")
    del sessions[session_id]
    return {"statut": "supprimé", "session_id": session_id}


@app.get("/sessions", tags=["Sessions"])
async def lister_sessions():
    """Liste toutes les sessions actives."""
    return {
        "sessions": [
            {"session_id": sid, "nb_messages": len(msgs)}
            for sid, msgs in sessions.items()
        ],
        "total": len(sessions),
    }


# ─── LANCEMENT DIRECT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("07_production.01_fastapi:app", host="0.0.0.0", port=8000, reload=True)
