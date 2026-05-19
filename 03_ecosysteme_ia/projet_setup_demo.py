"""
MODULE 3 — Projet Final : Demo Écosystème IA
=============================================
Ce projet assemble pydantic + httpx + loguru + rich + typer
pour créer un outil CLI complet qui :
  - Valide la config avec pydantic
  - Appelle l'API avec httpx
  - Logue tout avec loguru
  - Affiche les résultats avec rich
  - S'utilise en ligne de commande avec typer

Lance :
  python 03_ecosysteme_ia/projet_setup_demo.py --help
  python 03_ecosysteme_ia/projet_setup_demo.py check
  python 03_ecosysteme_ia/projet_setup_demo.py ask "Qu'est-ce qu'un agent IA ?"
  python 03_ecosysteme_ia/projet_setup_demo.py benchmark
"""

import asyncio
import os
import sys
import time
from pathlib import Path
import httpx
import typer
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

Path("03_ecosysteme_ia/data").mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", colorize=True)
logger.add("03_ecosysteme_ia/data/demo.log", level="DEBUG", rotation="1 MB")

console = Console()
app     = typer.Typer(help="[bold]Demo Écosystème IA — Module 3[/bold]", rich_markup_mode="rich")

MODELES = {
    "haiku":   "anthropic/claude-haiku-4-5",
    "sonnet":  "anthropic/claude-sonnet-4-5",
    "gpt4":    "openai/gpt-4o-mini",
    "mistral": "mistralai/mistral-small",
}


# ─── MODÈLES PYDANTIC ─────────────────────────────────────────────────────────

class Environnement(BaseModel):
    api_key:  str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout:  float = Field(default=30.0, ge=1.0, le=120.0)

    @field_validator("api_key")
    @classmethod
    def cle_non_vide(cls, v: str) -> str:
        if not v or not v.startswith("sk-"):
            raise ValueError("Clé API invalide (doit commencer par 'sk-')")
        return v


class ResultatBenchmark(BaseModel):
    modele:  str
    duree:   float
    tokens:  int
    extrait: str
    succes:  bool


# ─── LOGIQUE MÉTIER ───────────────────────────────────────────────────────────

def charger_env() -> Environnement | None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    try:
        return Environnement(api_key=api_key)
    except Exception as e:
        logger.error("Configuration invalide : {}", e)
        return None


async def appeler_modele(
    env: Environnement,
    modele: str,
    question: str,
    max_tokens: int = 300,
) -> ResultatBenchmark:
    debut = time.perf_counter()
    log = logger.bind(modele=modele)

    try:
        async with httpx.AsyncClient(
            base_url=env.base_url,
            headers={"Authorization": f"Bearer {env.api_key}"},
            timeout=env.timeout,
        ) as client:
            log.debug("Envoi de la requête")
            response = await client.post("/chat/completions", json={
                "model":      modele,
                "messages":   [{"role": "user", "content": question}],
                "max_tokens": max_tokens,
            })
            response.raise_for_status()

        data    = response.json()
        contenu = data["choices"][0]["message"]["content"] or ""
        tokens  = data.get("usage", {}).get("total_tokens", 0)
        duree   = time.perf_counter() - debut

        log.info("Réponse reçue en {:.2f}s ({} tokens)", duree, tokens)
        return ResultatBenchmark(modele=modele, duree=duree, tokens=tokens, extrait=contenu[:120], succes=True)

    except httpx.HTTPStatusError as e:
        duree = time.perf_counter() - debut
        log.error("Erreur HTTP {} : {}", e.response.status_code, e.response.text[:100])
        return ResultatBenchmark(modele=modele, duree=duree, tokens=0, extrait=f"HTTP {e.response.status_code}", succes=False)
    except Exception as e:
        duree = time.perf_counter() - debut
        log.error("Erreur : {}", e)
        return ResultatBenchmark(modele=modele, duree=duree, tokens=0, extrait=str(e)[:80], succes=False)


# ─── COMMANDES CLI ────────────────────────────────────────────────────────────

@app.command()
def check():
    """Vérifie que la clé API et la connexion fonctionnent."""
    env = charger_env()
    if not env:
        console.print(Panel("[red]OPENROUTER_API_KEY manquante ou invalide dans .env[/red]", title="Erreur"))
        raise typer.Exit(1)

    console.print(Panel(
        f"[green]✓[/green] Clé API détectée\n"
        f"[green]✓[/green] Base URL : {env.base_url}\n"
        f"[green]✓[/green] Timeout : {env.timeout}s",
        title="[bold green]Configuration OK[/bold green]",
    ))
    logger.info("Vérification terminée avec succès")


@app.command()
def ask(
    question:  str            = typer.Argument(..., help="Question à poser"),
    modele:    str            = typer.Option("haiku", "--modele", "-m", help="haiku|sonnet|gpt4|mistral"),
    max_tokens: int           = typer.Option(300, "--tokens", "-t", min=1, max=4096),
):
    """Pose une question à un LLM."""
    env = charger_env()
    if not env:
        raise typer.Exit(1)

    nom_modele = MODELES.get(modele, modele)
    logger.info("Question posée à {} : {!r}", nom_modele, question[:60])

    with console.status(f"[dim]Interrogation de {nom_modele}...[/dim]"):
        resultat = asyncio.run(appeler_modele(env, nom_modele, question, max_tokens))

    if resultat.succes:
        console.print(Panel(
            resultat.extrait,
            title=f"[cyan]{nom_modele}[/cyan] — {resultat.duree:.2f}s — {resultat.tokens} tokens",
        ))
    else:
        console.print(Panel(f"[red]{resultat.extrait}[/red]", title="Erreur"))
        raise typer.Exit(1)


@app.command()
def benchmark(
    question:  str = typer.Option("Qu'est-ce qu'un agent IA ? (1 phrase)", "--question", "-q"),
    max_tokens: int = typer.Option(100, "--tokens", "-t"),
):
    """Compare tous les modèles disponibles sur la même question."""
    env = charger_env()
    if not env:
        raise typer.Exit(1)

    logger.info("Benchmark de {} modèles", len(MODELES))
    console.print(f"\n[dim]Question : {question}[/dim]\n")

    with console.status("[dim]Interrogation en parallèle...[/dim]"):
        async def run_all():
            return await asyncio.gather(*[
                appeler_modele(env, modele, question, max_tokens)
                for modele in MODELES.values()
            ])
        resultats: list[ResultatBenchmark] = asyncio.run(run_all())

    # Affichage
    table = Table(title="Résultats du benchmark", show_lines=True)
    table.add_column("Modèle",   style="cyan",   min_width=30)
    table.add_column("Durée",    style="yellow", width=8)
    table.add_column("Tokens",   style="blue",   width=8)
    table.add_column("Réponse",  style="white",  min_width=40)

    tries = sorted(resultats, key=lambda r: r.duree)
    for i, r in enumerate(tries):
        statut = "[bold green]" if i == 0 else ""
        fin    = "[/bold green]" if i == 0 else ""
        extrait = r.extrait if r.succes else f"[red]{r.extrait}[/red]"
        table.add_row(
            f"{statut}{r.modele}{fin}",
            f"{statut}{r.duree:.2f}s{fin}",
            str(r.tokens) if r.tokens else "-",
            extrait,
        )

    console.print(table)
    valides = [r for r in resultats if r.succes]
    if valides:
        plus_rapide = min(valides, key=lambda r: r.duree)
        console.print(f"\n[bold green]Plus rapide :[/bold green] {plus_rapide.modele} ({plus_rapide.duree:.2f}s)")


if __name__ == "__main__":
    app()
