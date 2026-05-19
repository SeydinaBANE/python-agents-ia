"""
MODULE 4 — Leçon 1 : Messages API
====================================
C'est la base de tout : comment envoyer des messages à un LLM
et structurer une conversation multi-tours.

L'API est identique sur OpenRouter, OpenAI, et Anthropic (via compatibilité).
"""

import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# ─── CLIENT PARTAGÉ ───────────────────────────────────────────────────────────

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

MODELE = "anthropic/claude-haiku-4-5"


# ─── 1. APPEL SIMPLE ──────────────────────────────────────────────────────────

async def appel_simple(question: str) -> str:
    client = get_client()
    response = await client.chat.completions.create(
        model=MODELE,
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


# ─── 2. SYSTEM PROMPT — donner une identité à l'agent ─────────────────────────
# Le system prompt définit le comportement, le ton, et les contraintes de l'agent.
# Il est envoyé à chaque appel, avant tous les messages utilisateur.

async def avec_system_prompt(question: str, system: str) -> str:
    client = get_client()
    response = await client.chat.completions.create(
        model=MODELE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


# ─── 3. CONVERSATION MULTI-TOURS ──────────────────────────────────────────────
# Le LLM est sans mémoire : il faut lui renvoyer TOUT l'historique à chaque appel.
# C'est ce qu'on appelle le "context window".

class Conversation:
    """Gère une conversation avec un LLM en conservant l'historique."""

    def __init__(self, system: str = "Tu es un assistant IA utile et concis."):
        self.system   = system
        self.messages: list[dict] = []
        self.client   = get_client()

    async def envoyer(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})

        response = await self.client.chat.completions.create(
            model=MODELE,
            messages=[{"role": "system", "content": self.system}] + self.messages,
            max_tokens=512,
        )

        reponse = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reponse})
        return reponse

    @property
    def nb_messages(self) -> int:
        return len(self.messages)

    @property
    def tokens_approx(self) -> int:
        total = sum(len(m["content"]) for m in self.messages)
        return total // 4

    def reset(self) -> None:
        self.messages.clear()


# ─── 4. PARAMÈTRES IMPORTANTS ─────────────────────────────────────────────────

async def demo_parametres():
    client = get_client()

    # temperature : 0.0 = déterministe / précis, 2.0 = créatif / aléatoire
    # Pour les agents : 0.0 à 0.3 (on veut des réponses fiables)
    # Pour la créativité : 0.7 à 1.2

    responses = await asyncio.gather(
        client.chat.completions.create(
            model=MODELE,
            messages=[{"role": "user", "content": "Donne-moi un mot au hasard."}],
            max_tokens=10,
            temperature=0.0,    # toujours le même mot
        ),
        client.chat.completions.create(
            model=MODELE,
            messages=[{"role": "user", "content": "Donne-moi un mot au hasard."}],
            max_tokens=10,
            temperature=1.5,    # mot différent à chaque fois
        ),
    )

    console.print(f"temperature=0.0 : {responses[0].choices[0].message.content}")
    console.print(f"temperature=1.5 : {responses[1].choices[0].message.content}")


# ─── 5. INSPECTER LA RÉPONSE COMPLÈTE ────────────────────────────────────────
# L'objet response contient bien plus que le texte

async def inspecter_reponse():
    client = get_client()
    response = await client.chat.completions.create(
        model=MODELE,
        messages=[{"role": "user", "content": "Bonjour !"}],
        max_tokens=50,
    )

    console.print(f"\n[bold]id[/bold]            : {response.id}")
    console.print(f"[bold]modèle[/bold]        : {response.model}")
    console.print(f"[bold]stop_reason[/bold]   : {response.choices[0].finish_reason}")
    console.print(f"[bold]contenu[/bold]       : {response.choices[0].message.content}")

    if response.usage:
        console.print(f"[bold]tokens prompt[/bold] : {response.usage.prompt_tokens}")
        console.print(f"[bold]tokens réponse[/bold]: {response.usage.completion_tokens}")
        console.print(f"[bold]tokens total[/bold]  : {response.usage.total_tokens}")


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 1 — Messages API[/bold]")

    # Test 1 : appel simple
    console.print("\n[bold cyan]1. Appel simple[/bold cyan]")
    rep = await appel_simple("Qu'est-ce qu'un LLM en une phrase ?")
    console.print(rep)

    # Test 2 : system prompt
    console.print("\n[bold cyan]2. Avec system prompt[/bold cyan]")
    rep = await avec_system_prompt(
        "Qu'est-ce que Python ?",
        system="Tu es un professeur qui explique les concepts techniques à des débutants avec des analogies simples. Réponds en 2 phrases max.",
    )
    console.print(rep)

    # Test 3 : conversation multi-tours
    console.print("\n[bold cyan]3. Conversation multi-tours[/bold cyan]")
    conv = Conversation(system="Tu es un expert Python. Réponds de façon concise.")
    r1 = await conv.envoyer("Qu'est-ce qu'une liste en Python ?")
    console.print(f"[green]Bot:[/green] {r1}")
    r2 = await conv.envoyer("Quelle est la différence avec un tuple ?")
    console.print(f"[green]Bot:[/green] {r2}")
    r3 = await conv.envoyer("Et un dict ?")
    console.print(f"[green]Bot:[/green] {r3}")
    console.print(f"[dim]{conv.nb_messages} messages, ~{conv.tokens_approx} tokens[/dim]")

    # Test 4 : inspecter la réponse
    console.print("\n[bold cyan]4. Inspecter la réponse[/bold cyan]")
    await inspecter_reponse()

    # Test 5 : paramètres
    console.print("\n[bold cyan]5. Effet de temperature[/bold cyan]")
    await demo_parametres()


if __name__ == "__main__":
    asyncio.run(main())
