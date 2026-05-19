"""
MODULE 3 — Leçon 2 : HTTPX
============================
httpx est le client HTTP moderne de Python — il supporte async natif,
les timeouts, la gestion d'erreurs propre, et les retries.
C'est ce qui tourne sous le capot de la plupart des SDK LLM.
"""

import asyncio
import httpx
from pydantic import BaseModel

# ─── REQUÊTE SYNCHRONE SIMPLE ────────────────────────────────────────────────

def test_sync():
    # with → ferme la connexion automatiquement
    with httpx.Client(timeout=10.0) as client:
        response = client.get("https://httpbin.org/json")
        print(response.status_code)            # 200
        print(response.headers["content-type"])
        data = response.json()
        print(data)

# test_sync()  # décommente pour tester


# ─── REQUÊTE ASYNC ────────────────────────────────────────────────────────────

async def test_async():
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get("https://httpbin.org/json")
        print(response.status_code)
        return response.json()

# asyncio.run(test_async())


# ─── CONFIGURATION RÉUTILISABLE ───────────────────────────────────────────────
# Dans les agents, on crée un client configuré une fois et on le réutilise

def creer_client_openrouter(api_key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        timeout=httpx.Timeout(
            connect=5.0,    # temps max pour établir la connexion
            read=60.0,      # temps max pour lire la réponse (LLMs peuvent être lents)
            write=10.0,
            pool=5.0,
        ),
    )


# ─── APPEL À L'API OPENROUTER AVEC HTTPX BRUT ─────────────────────────────────
# Sans SDK — utile pour comprendre ce qui se passe vraiment

import os
from dotenv import load_dotenv
load_dotenv()

async def appel_llm_httpx(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    payload = {
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }

    async with creer_client_openrouter(api_key) as client:
        response = await client.post("/chat/completions", json=payload)

        # Toujours vérifier le statut avant de parser
        response.raise_for_status()     # lève HTTPStatusError si 4xx/5xx

        data = response.json()
        return data["choices"][0]["message"]["content"]


# ─── GESTION DES ERREURS HTTP ─────────────────────────────────────────────────

async def appel_robuste(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    try:
        async with creer_client_openrouter(api_key) as client:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": "anthropic/claude-haiku-4-5",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    except httpx.TimeoutException:
        return "[Erreur] Timeout — le serveur n'a pas répondu à temps"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "[Erreur] Clé API invalide"
        if e.response.status_code == 429:
            return "[Erreur] Rate limit dépassé — attends quelques secondes"
        return f"[Erreur HTTP {e.response.status_code}] {e.response.text[:200]}"
    except httpx.ConnectError:
        return "[Erreur] Impossible de se connecter à OpenRouter"


# ─── PLUSIEURS REQUÊTES EN PARALLÈLE ──────────────────────────────────────────

async def requetes_paralleles(urls: list[str]) -> list[dict]:
    """Télécharge plusieurs URLs simultanément."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        taches = [client.get(url) for url in urls]
        reponses = await asyncio.gather(*taches, return_exceptions=True)

    resultats = []
    for url, rep in zip(urls, reponses):
        if isinstance(rep, Exception):
            resultats.append({"url": url, "succes": False, "erreur": str(rep)})
        else:
            resultats.append({
                "url":    url,
                "succes": True,
                "statut": rep.status_code,
                "taille": len(rep.content),
            })
    return resultats


# ─── STREAMING AVEC HTTPX ────────────────────────────────────────────────────
# Les LLMs streamant des Server-Sent Events (SSE)
# httpx.aiter_lines() lit ligne par ligne sans attendre la fin

async def stream_httpx(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    payload = {
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "stream": True,
    }
    reponse_complete = ""
    print("Bot : ", end="", flush=True)

    async with creer_client_openrouter(api_key) as client:
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            async for ligne in response.aiter_lines():
                if not ligne.startswith("data: "):
                    continue
                contenu = ligne[6:]             # retire "data: "
                if contenu == "[DONE]":
                    break
                import json
                chunk = json.loads(contenu)
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    print(token, end="", flush=True)
                    reponse_complete += token

    print()
    return reponse_complete


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    from rich.console import Console
    console = Console()

    # Test appel direct
    console.print("\n[bold]Test appel HTTPX direct :[/bold]")
    reponse = await appel_robuste("Qu'est-ce que httpx en Python ? (1 phrase)")
    console.print(f"[green]{reponse}[/green]")

    # Test streaming
    console.print("\n[bold]Test streaming HTTPX :[/bold]")
    await stream_httpx("Explique les avantages d'httpx en 2 phrases.")


if __name__ == "__main__":
    asyncio.run(main())
