"""
MODULE 2 — Projet Final : Client Async Multi-Modèles
=====================================================
Ce projet combine POO + décorateurs + async/await + type hints.

Il interroge 3 modèles LLM différents EN PARALLÈLE sur la même question,
mesure les temps de réponse, et affiche un tableau comparatif.

Lance : python 02_intermediaire/projet_async_client.py
"""

import asyncio
import time
import os
from typing import TypedDict
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()

console = Console()


# ─── TYPES ────────────────────────────────────────────────────────────────────

class ResultatModele(TypedDict):
    modele: str
    reponse: str
    duree: float
    tokens: int
    succes: bool
    erreur: str | None


# ─── CLIENT LLM ASYNC (POO + async) ──────────────────────────────────────────

@dataclass
class ClientMultiModeles:
    """Interroge plusieurs modèles LLM en parallèle via OpenRouter."""

    modeles: list[str] = field(default_factory=lambda: [
        "anthropic/claude-haiku-4-5",
        "openai/gpt-4o-mini",
        "mistralai/mistral-small",
    ])
    max_tokens: int = 512
    timeout: float = 30.0

    def __post_init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY manquante dans .env")
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def interroger_un_modele(
        self,
        modele: str,
        messages: list[dict],
    ) -> ResultatModele:
        """Interroge un seul modèle et retourne le résultat typé."""
        debut = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout):
                response = await self._client.chat.completions.create(
                    model=modele,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
            duree = time.perf_counter() - debut
            return {
                "modele": modele,
                "reponse": response.choices[0].message.content or "",
                "duree": duree,
                "tokens": response.usage.total_tokens if response.usage else 0,
                "succes": True,
                "erreur": None,
            }
        except asyncio.TimeoutError:
            return {
                "modele": modele,
                "reponse": "",
                "duree": self.timeout,
                "tokens": 0,
                "succes": False,
                "erreur": f"Timeout après {self.timeout}s",
            }
        except Exception as e:
            return {
                "modele": modele,
                "reponse": "",
                "duree": time.perf_counter() - debut,
                "tokens": 0,
                "succes": False,
                "erreur": str(e)[:100],
            }

    async def interroger_tous(
        self,
        question: str,
        system: str = "Réponds en français, de manière concise.",
    ) -> list[ResultatModele]:
        """Interroge tous les modèles EN PARALLÈLE."""
        messages = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": question},
        ]
        debut_global = time.perf_counter()

        resultats = await asyncio.gather(
            *[self.interroger_un_modele(m, messages) for m in self.modeles],
            return_exceptions=False,
        )

        duree_totale = time.perf_counter() - debut_global
        console.print(f"\n[dim]{len(self.modeles)} modèles interrogés en {duree_totale:.2f}s[/dim]")

        return list(resultats)

    async def stream_meilleur(self, question: str) -> None:
        """Streame la réponse du modèle le plus rapide."""
        messages = [{"role": "user", "content": question}]
        modele = self.modeles[0]

        console.print(f"\n[bold]Streaming depuis {modele} :[/bold]")
        print("Bot : ", end="", flush=True)

        async with await self._client.chat.completions.create(
            model=modele,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
        ) as stream:
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                print(token, end="", flush=True)
        print()


# ─── AFFICHAGE (rich) ─────────────────────────────────────────────────────────

def afficher_resultats(question: str, resultats: list[ResultatModele]) -> None:
    table = Table(title=f"Question : {question[:60]}{'...' if len(question) > 60 else ''}")
    table.add_column("Modèle",    style="cyan",   min_width=30)
    table.add_column("Statut",    style="green",  width=8)
    table.add_column("Durée",     style="yellow", width=7)
    table.add_column("Tokens",    style="blue",   width=7)
    table.add_column("Réponse",   style="white",  min_width=40)

    for r in sorted(resultats, key=lambda x: x["duree"]):
        statut  = "✓" if r["succes"] else "✗"
        duree   = f"{r['duree']:.2f}s"
        tokens  = str(r["tokens"]) if r["tokens"] else "-"
        extrait = (r["reponse"][:80] + "…") if len(r["reponse"]) > 80 else r["reponse"]
        if not r["succes"]:
            extrait = f"[red]{r['erreur']}[/red]"
            statut  = "[red]✗[/red]"
        table.add_row(r["modele"], statut, duree, tokens, extrait)

    console.print(table)


# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────

async def main() -> None:
    console.print(Panel(
        "[bold green]Client Multi-Modèles Async — Module 2[/bold green]\n"
        "Compare 3 LLMs en parallèle sur la même question.\n"
        "Tape [cyan]stream[/cyan] pour le mode streaming · [cyan]quitter[/cyan] pour sortir",
        title="Bienvenue",
    ))

    client = ClientMultiModeles()

    while True:
        try:
            question = Prompt.ask("\n[bold blue]Question[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]À bientôt ![/bold]")
            break

        if not question:
            continue

        if question.lower() == "quitter":
            console.print("[bold]À bientôt ![/bold]")
            break

        if question.lower() == "stream":
            q = Prompt.ask("Question à streamer").strip()
            await client.stream_meilleur(q)
            continue

        with console.status("[dim]Interrogation des modèles en parallèle...[/dim]"):
            resultats = await client.interroger_tous(question)

        afficher_resultats(question, resultats)


if __name__ == "__main__":
    asyncio.run(main())
