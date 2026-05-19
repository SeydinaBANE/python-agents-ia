"""
MODULE 4 — Leçon 2 : Streaming
================================
Le streaming affiche les tokens au fur et à mesure au lieu d'attendre
la réponse complète. C'est ce que fait ChatGPT.

Avantages pour les agents :
- L'utilisateur voit la réponse se construire (meilleure UX)
- Tu peux détecter tôt si l'agent part dans la mauvaise direction
- Tu peux interrompre un appel si nécessaire
"""

import asyncio
import os
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

load_dotenv()
console = Console()
MODELE = "anthropic/claude-haiku-4-5"

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


# ─── 1. STREAM BASIQUE ────────────────────────────────────────────────────────

async def stream_simple(prompt: str) -> str:
    """Affiche les tokens en temps réel et retourne la réponse complète."""
    client = get_client()
    reponse = ""

    print("Bot : ", end="", flush=True)

    # stream=True → retourne un AsyncStream qu'on itère avec async for
    async with await client.chat.completions.create(
        model=MODELE,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        stream=True,
    ) as stream:
        async for chunk in stream:
            # Chaque chunk contient un delta (fragment de texte)
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            reponse += token

    print()  # saut de ligne final
    return reponse


# ─── 2. STREAM AVEC RICH LIVE ─────────────────────────────────────────────────
# Rich Live permet de mettre à jour l'affichage en place
# Parfait pour rendre le Markdown au fur et à mesure

async def stream_avec_markdown(prompt: str) -> str:
    """Stream avec rendu Markdown en temps réel."""
    client = get_client()
    reponse = ""

    with Live(console=console, refresh_per_second=15) as live:
        async with await client.chat.completions.create(
            model=MODELE,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            stream=True,
        ) as stream:
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                reponse += token
                # Mise à jour du rendu Markdown à chaque token
                live.update(Markdown(reponse))

    return reponse


# ─── 3. STREAM AVEC MESURE DE PERFORMANCE ────────────────────────────────────

async def stream_avec_stats(prompt: str) -> dict:
    """Stream + mesure du Time to First Token (TTFT) et débit."""
    client = get_client()
    reponse = ""
    tokens_recus = 0
    premier_token_recu = False
    ttft = 0.0
    debut = time.perf_counter()

    print("Bot : ", end="", flush=True)

    async with await client.chat.completions.create(
        model=MODELE,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True,
    ) as stream:
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                if not premier_token_recu:
                    ttft = time.perf_counter() - debut  # Time to First Token
                    premier_token_recu = True
                print(token, end="", flush=True)
                reponse += token
                tokens_recus += 1

    duree_totale = time.perf_counter() - debut
    print()

    return {
        "reponse":      reponse,
        "tokens":       tokens_recus,
        "ttft_s":       round(ttft, 3),
        "duree_totale": round(duree_totale, 3),
        "tokens_par_s": round(tokens_recus / duree_totale, 1) if duree_totale > 0 else 0,
    }


# ─── 4. STREAM D'UNE CONVERSATION ────────────────────────────────────────────

async def stream_conversation(historique: list[dict], nouveau_message: str) -> str:
    """Envoie un message dans une conversation existante et streame la réponse."""
    client = get_client()

    messages = historique + [{"role": "user", "content": nouveau_message}]
    reponse = ""

    print(f"\n[Bot] ", end="", flush=True)
    async with await client.chat.completions.create(
        model=MODELE,
        messages=messages,
        max_tokens=512,
        stream=True,
    ) as stream:
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            reponse += token
    print()

    return reponse


# ─── 5. STREAM AVEC INTERRUPTION ─────────────────────────────────────────────
# On peut arrêter le stream dès qu'une condition est remplie

async def stream_jusqu_a(prompt: str, stop_pattern: str) -> str:
    """Interrompt le stream si le stop_pattern est détecté."""
    client = get_client()
    reponse = ""

    print("Bot (avec interruption possible) : ", end="", flush=True)

    async with await client.chat.completions.create(
        model=MODELE,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=True,
    ) as stream:
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            reponse += token
            # Interrompre si le pattern apparaît
            if stop_pattern.lower() in reponse.lower():
                print(" [interrompu]", flush=True)
                break

    print()
    return reponse


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 2 — Streaming[/bold]")

    # Test 1 : stream basique
    console.print("\n[bold cyan]1. Stream basique[/bold cyan]")
    await stream_simple("Explique asyncio en Python en 3 points clés.")

    # Test 2 : stream avec rendu Markdown
    console.print("\n[bold cyan]2. Stream avec Markdown[/bold cyan]")
    await stream_avec_markdown(
        "Donne-moi un exemple de code Python pour appeler une API avec httpx. "
        "Formate ta réponse en Markdown avec un bloc de code."
    )

    # Test 3 : stats de performance
    console.print("\n[bold cyan]3. Mesure des performances[/bold cyan]")
    stats = await stream_avec_stats("Qu'est-ce qu'un agent IA ? Réponds en 5 phrases.")
    console.print(f"\n[dim]TTFT: {stats['ttft_s']}s | Durée: {stats['duree_totale']}s | Débit: {stats['tokens_par_s']} tokens/s[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
