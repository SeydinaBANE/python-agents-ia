"""
MODULE 2 — Leçon 5 : Async / Await  ← LA leçon la plus importante
===================================================================
Les agents IA font des dizaines d'appels réseau (LLM, outils, APIs).
Avec du code synchrone, ils attendent l'un après l'autre.
Avec async/await, ils tournent EN PARALLÈLE → 5x plus rapide.

Règle d'or :
  - async def  → déclare une coroutine (fonction asynchrone)
  - await      → "attends ce résultat sans bloquer les autres"
  - asyncio.run() → point d'entrée pour lancer le code async
"""

import asyncio
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ─── ANALOGIE CONCRÈTE ────────────────────────────────────────────────────────
# Synchrone  = un seul cuisinier qui fait : café → toast → jus → omelette (séquentiellement)
# Asynchrone = un cuisinier qui lance tout en même temps et surveille


# ─── 1. COROUTINE SIMPLE ──────────────────────────────────────────────────────

async def saluer(nom: str, delai: float) -> str:
    print(f"→ Début salutation pour {nom}")
    await asyncio.sleep(delai)          # simule un appel réseau
    print(f"← Fin salutation pour {nom}")
    return f"Bonjour {nom} !"


# asyncio.run() lance la boucle d'événements et exécute la coroutine
resultat = asyncio.run(saluer("Seydina", 0.1))
print(resultat)


# ─── 2. SÉQUENTIEL VS PARALLÈLE ───────────────────────────────────────────────

async def demo_sequentiel():
    debut = time.perf_counter()
    r1 = await saluer("Alice", 0.2)     # attend 0.2s
    r2 = await saluer("Bob",   0.2)     # attend encore 0.2s
    r3 = await saluer("Charlie", 0.2)   # attend encore 0.2s
    print(f"Séquentiel : {time.perf_counter() - debut:.2f}s")  # ≈ 0.6s
    return [r1, r2, r3]

async def demo_parallele():
    debut = time.perf_counter()
    # asyncio.gather() lance TOUTES les coroutines en même temps
    resultats = await asyncio.gather(
        saluer("Alice", 0.2),
        saluer("Bob",   0.2),
        saluer("Charlie", 0.2),
    )
    print(f"Parallèle : {time.perf_counter() - debut:.2f}s")   # ≈ 0.2s !
    return list(resultats)

print("\n--- Séquentiel ---")
asyncio.run(demo_sequentiel())

print("\n--- Parallèle ---")
asyncio.run(demo_parallele())


# ─── 3. APPEL RÉEL AU LLM EN ASYNC ───────────────────────────────────────────

async def interroger_llm(question: str, modele: str = "anthropic/claude-haiku-4-5") -> str:
    """Pose une question à un LLM et retourne la réponse."""
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    response = await client.chat.completions.create(
        model=modele,
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    return response.choices[0].message.content or ""


# ─── 4. PLUSIEURS QUESTIONS EN PARALLÈLE ──────────────────────────────────────

async def recherche_multi_questions(questions: list[str]) -> list[dict]:
    """
    Pose plusieurs questions au LLM en parallèle.
    Utile pour un agent qui doit synthétiser plusieurs sources.
    """
    debut = time.perf_counter()

    # Lance tous les appels LLM simultanément
    reponses = await asyncio.gather(
        *[interroger_llm(q) for q in questions],
        return_exceptions=True,     # si un appel échoue, ne plante pas tout
    )

    duree = time.perf_counter() - debut
    print(f"\n{len(questions)} questions en {duree:.2f}s (parallèle)")

    resultats = []
    for question, reponse in zip(questions, reponses):
        if isinstance(reponse, Exception):
            resultats.append({"question": question, "reponse": f"Erreur : {reponse}", "ok": False})
        else:
            resultats.append({"question": question, "reponse": reponse, "ok": True})

    return resultats


# ─── 5. STREAMING ASYNC ───────────────────────────────────────────────────────

async def stream_reponse(prompt: str) -> str:
    """Affiche les tokens au fur et à mesure (comme ChatGPT)."""
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    reponse_complete = ""
    print("Bot : ", end="", flush=True)

    # stream=True → retourne un AsyncIterator de chunks
    async with await client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True,
    ) as stream:
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            reponse_complete += token

    print()  # saut de ligne
    return reponse_complete


# ─── 6. ASYNCIO.TIMEOUT — ne pas attendre indéfiniment ───────────────────────

async def appel_avec_timeout(prompt: str, timeout: float = 10.0) -> str:
    try:
        async with asyncio.timeout(timeout):
            return await interroger_llm(prompt)
    except asyncio.TimeoutError:
        return f"[Timeout après {timeout}s]"


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    # Test 1 : plusieurs questions en parallèle
    questions = [
        "Qu'est-ce qu'un agent IA ? (1 phrase)",
        "Qu'est-ce qu'asyncio en Python ? (1 phrase)",
        "Qu'est-ce qu'un LLM ? (1 phrase)",
    ]

    console.print(Panel("[bold]Test : 3 questions en parallèle[/bold]"))
    resultats = await recherche_multi_questions(questions)
    for r in resultats:
        console.print(f"\n[cyan]Q:[/cyan] {r['question']}")
        console.print(f"[green]R:[/green] {r['reponse'][:150]}")

    # Test 2 : streaming
    console.print(Panel("[bold]Test : streaming[/bold]"))
    await stream_reponse("Explique async/await en Python en 2 phrases simples.")


if __name__ == "__main__":
    asyncio.run(main())
