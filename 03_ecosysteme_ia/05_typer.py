"""
MODULE 3 — Leçon 5 : Typer
============================
Typer crée des CLI professionnelles à partir de tes fonctions Python.
Tu décris les arguments avec les type hints — Typer génère l'aide,
la validation, et l'autocomplétion automatiquement.

Lance : python 03_ecosysteme_ia/05_typer.py --help
"""

import typer
from typing import Optional
from enum import Enum
from pathlib import Path
from rich.console import Console
from rich.table import Table

app    = typer.Typer(help="Outils CLI pour les agents IA", rich_markup_mode="rich")
console = Console()


# ─── COMMANDE SIMPLE ──────────────────────────────────────────────────────────

@app.command()
def saluer(
    nom:    str           = typer.Argument(..., help="Ton prénom"),
    majuscule: bool       = typer.Option(False, "--majuscule", "-m", help="Mettre en majuscules"),
):
    """Affiche un message de bienvenue."""
    message = f"Bonjour {nom} !"
    console.print(f"[bold green]{message.upper() if majuscule else message}[/bold green]")


# ─── COMMANDE AVEC ENUM (choix restreints) ────────────────────────────────────

class Modele(str, Enum):
    haiku   = "anthropic/claude-haiku-4-5"
    sonnet  = "anthropic/claude-sonnet-4-5"
    gpt4    = "openai/gpt-4o-mini"
    mistral = "mistralai/mistral-small"


@app.command()
def chat(
    question: str               = typer.Argument(..., help="Question à poser au LLM"),
    modele:   Modele            = typer.Option(Modele.haiku, "--modele", "-m", help="Modèle LLM"),
    tokens:   int               = typer.Option(512, "--tokens", "-t", help="Max tokens", min=1, max=4096),
    verbose:  bool              = typer.Option(False, "--verbose", "-v", help="Mode verbeux"),
):
    """
    Pose une question à un LLM via OpenRouter.

    [bold]Exemples :[/bold]

        python 03_ecosysteme_ia/05_typer.py chat "Bonjour !"
        python 03_ecosysteme_ia/05_typer.py chat "Résume Python" --modele sonnet --tokens 1024
    """
    import asyncio, os
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    load_dotenv()

    if verbose:
        console.print(f"[dim]Modèle : {modele.value}[/dim]")
        console.print(f"[dim]Max tokens : {tokens}[/dim]")

    async def appeler():
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        with console.status("[dim]Appel en cours...[/dim]"):
            response = await client.chat.completions.create(
                model=modele.value,
                messages=[{"role": "user", "content": question}],
                max_tokens=tokens,
            )
        return response.choices[0].message.content or ""

    reponse = asyncio.run(appeler())
    console.print(f"\n[bold green]Réponse :[/bold green]\n{reponse}")


# ─── COMMANDE AVEC FICHIER ────────────────────────────────────────────────────

@app.command()
def historique(
    fichier: Path = typer.Option(
        Path("01_fondations/data/chatbot_historique.json"),
        "--fichier", "-f",
        help="Chemin vers le fichier JSON d'historique",
        exists=False,
    ),
    effacer: bool = typer.Option(False, "--effacer", help="Effacer l'historique"),
):
    """Affiche ou efface l'historique d'une conversation."""
    import json

    if effacer:
        if fichier.exists():
            fichier.unlink()
            console.print(f"[yellow]Historique effacé : {fichier}[/yellow]")
        else:
            console.print("[dim]Aucun historique à effacer.[/dim]")
        return

    if not fichier.exists():
        console.print(f"[dim]Aucun historique trouvé : {fichier}[/dim]")
        raise typer.Exit(code=0)

    try:
        msgs = json.loads(fichier.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        console.print("[red]Fichier JSON invalide[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Historique — {fichier.name} ({len(msgs)} messages)")
    table.add_column("#",       style="dim",   width=4)
    table.add_column("Rôle",    style="cyan",  width=12)
    table.add_column("Message", style="white", min_width=50)

    for i, msg in enumerate(msgs, 1):
        role    = msg.get("role", "?")
        contenu = msg.get("content", "")[:100]
        couleur = "bold green" if role == "assistant" else "bold blue"
        table.add_row(str(i), f"[{couleur}]{role}[/{couleur}]", contenu)

    console.print(table)


# ─── GROUPE DE COMMANDES (sous-CLI) ───────────────────────────────────────────
# Pour organiser une CLI complexe en sous-commandes :
# python cli.py agent start ...
# python cli.py agent list ...

agent_app = typer.Typer(help="Gérer les agents")
app.add_typer(agent_app, name="agent")

@agent_app.command("list")
def agent_list():
    """Liste les agents configurés."""
    agents = [
        {"nom": "Atlas",  "modele": "claude-haiku",  "statut": "actif"},
        {"nom": "Nova",   "modele": "gpt-4o-mini",   "statut": "inactif"},
        {"nom": "Orion",  "modele": "mistral-small", "statut": "actif"},
    ]
    table = Table(title="Agents disponibles")
    table.add_column("Nom",     style="cyan")
    table.add_column("Modèle",  style="yellow")
    table.add_column("Statut",  style="green")

    for a in agents:
        statut_style = "bold green" if a["statut"] == "actif" else "dim red"
        table.add_row(a["nom"], a["modele"], f"[{statut_style}]{a['statut']}[/{statut_style}]")

    console.print(table)

@agent_app.command("create")
def agent_create(
    nom:    str    = typer.Argument(..., help="Nom de l'agent"),
    modele: Modele = typer.Option(Modele.haiku, "--modele", "-m"),
):
    """Crée un nouvel agent."""
    console.print(f"[bold green]Agent '{nom}' créé avec le modèle {modele.value}[/bold green]")


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
