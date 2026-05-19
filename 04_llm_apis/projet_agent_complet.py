"""
MODULE 4 — Projet Final : Agent Complet
=========================================
Cet agent réunit TOUT le Module 4 :
- Messages API multi-tours       (leçon 1)
- Streaming en temps réel        (leçon 2)
- Tool use avec boucle agent     (leçon 3)
- Structured output (Pydantic)   (leçon 4)
- Prompt engineering             (leçon 5)

C'est un agent conversationnel avec mémoire, outils réels,
réponses streamées et logging complet.

Lance : python 04_llm_apis/projet_agent_complet.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()
console = Console()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

MODELE        = "anthropic/claude-sonnet-4-5"
HISTORIQUE_PATH = Path("04_llm_apis/data/historique.json")
Path("04_llm_apis/data").mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add("04_llm_apis/data/agent.log", level="DEBUG", rotation="1 MB")

SYSTEM_PROMPT = """Tu es Atlas, un assistant IA expert en Python et en développement d'agents IA.

## Tes outils disponibles
- get_meteo(ville) : météo d'une ville
- calculer(expression) : calculs mathématiques
- chercher_info(sujet, nb_resultats) : recherche d'informations
- heure_actuelle() : date et heure actuelles

## Règles
- Réponds en français, de façon concise et précise
- Utilise les outils quand c'est pertinent, pas systématiquement
- Pour les calculs, utilise TOUJOURS l'outil (évite les erreurs)
- Après un outil, synthétise le résultat naturellement
- Si tu ne sais pas quelque chose, dis-le clairement"""


# ─── OUTILS ───────────────────────────────────────────────────────────────────

OUTILS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_meteo",
            "description": "Retourne la météo actuelle d'une ville.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ville": {"type": "string", "description": "Nom de la ville"},
                },
                "required": ["ville"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculer",
            "description": "Évalue une expression mathématique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Ex: '15 * 8 + 3'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chercher_info",
            "description": "Cherche des informations sur un sujet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sujet":        {"type": "string"},
                    "nb_resultats": {"type": "integer", "default": 3},
                },
                "required": ["sujet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heure_actuelle",
            "description": "Retourne la date et l'heure actuelles.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def get_meteo(ville: str) -> dict:
    meteos = {
        "Paris":  {"temp": "18°C", "condition": "nuageux"},
        "Dakar":  {"temp": "32°C", "condition": "ensoleillé"},
        "Lyon":   {"temp": "15°C", "condition": "pluvieux"},
        "Tokyo":  {"temp": "22°C", "condition": "brumeux"},
        "New York": {"temp": "25°C", "condition": "ensoleillé"},
    }
    return meteos.get(ville, {"temp": "20°C", "condition": "inconnu"}) | {"ville": ville}

def calculer(expression: str) -> dict:
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Expression non autorisée"}
        return {"expression": expression, "resultat": eval(expression)}  # noqa: S307
    except Exception as e:
        return {"erreur": str(e)}

def chercher_info(sujet: str, nb_resultats: int = 3) -> dict:
    return {
        "sujet": sujet,
        "resultats": [f"Info #{i+1} sur '{sujet}'" for i in range(nb_resultats)],
    }

def heure_actuelle() -> dict:
    return {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


REGISTRE = {
    "get_meteo":    get_meteo,
    "calculer":     calculer,
    "chercher_info": chercher_info,
    "heure_actuelle": heure_actuelle,
}


# ─── GESTION HISTORIQUE ───────────────────────────────────────────────────────

def charger_historique() -> list[dict]:
    if not HISTORIQUE_PATH.exists():
        return []
    try:
        return json.loads(HISTORIQUE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

def sauvegarder_historique(historique: list[dict]) -> None:
    # On ne sauvegarde que les messages user/assistant (pas les tool_calls internes)
    a_sauvegarder = [m for m in historique if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)]
    HISTORIQUE_PATH.write_text(
        json.dumps(a_sauvegarder, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─── BOUCLE AGENT AVEC STREAMING ──────────────────────────────────────────────

async def executer_tour(client: AsyncOpenAI, messages: list[dict]) -> str:
    """
    Un tour complet : LLM → tools → LLM → réponse finale streamée.
    Retourne la réponse finale en texte.
    """
    iteration     = 0
    reponse_finale = ""

    while True:
        iteration += 1

        # ── Appel LLM ──────────────────────────────────────────────────────
        response = await client.chat.completions.create(
            model=MODELE,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=OUTILS_SCHEMA,
            tool_choice="auto",
            max_tokens=1024,
        )

        choix = response.choices[0]

        # ── Réponse finale — on la streame ──────────────────────────────────
        if choix.finish_reason == "stop":
            contenu = choix.message.content or ""

            # Re-appel avec stream=True pour afficher token par token
            reponse_finale = ""
            with Live(console=console, refresh_per_second=15) as live:
                async with await client.chat.completions.create(
                    model=MODELE,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    max_tokens=1024,
                    stream=True,
                ) as stream:
                    async for chunk in stream:
                        token = chunk.choices[0].delta.content or ""
                        reponse_finale += token
                        live.update(Markdown(reponse_finale))

            return reponse_finale

        # ── Tool calls ───────────────────────────────────────────────────────
        if choix.finish_reason == "tool_calls":
            messages.append(choix.message)

            for tc in choix.message.tool_calls:
                nom  = tc.function.name
                args = json.loads(tc.function.arguments)

                console.print(f"  [yellow]⚙ {nom}[/yellow]({args})")
                logger.debug("Tool call : {}({})", nom, args)

                debut = time.perf_counter()
                resultat = REGISTRE.get(nom, lambda **_: {"erreur": "inconnu"})(**args)
                duree = time.perf_counter() - debut

                logger.debug("Résultat {} en {:.3f}s : {}", nom, duree, resultat)
                console.print(f"  [dim]→ {resultat}[/dim]")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(resultat, ensure_ascii=False),
                })

        if iteration >= 8:
            return "[Erreur] Trop d'itérations"


# ─── INTERFACE CLI ────────────────────────────────────────────────────────────

async def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print(Panel("[red]OPENROUTER_API_KEY manquante dans .env[/red]", title="Erreur"))
        sys.exit(1)

    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    historique = charger_historique()

    console.print(Panel(
        "[bold green]Agent Atlas — Module 4[/bold green]\n"
        f"Modèle : [cyan]{MODELE}[/cyan]\n"
        "Commandes : [cyan]/tools[/cyan] [cyan]/reset[/cyan] [cyan]/quitter[/cyan]",
        title="Bienvenue",
    ))
    if historique:
        console.print(f"[dim]{len(historique)} messages précédents chargés.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]Toi[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            sauvegarder_historique(historique)
            console.print("\n[bold]À bientôt ![/bold]")
            break

        if not user_input:
            continue

        if user_input == "/tools":
            noms = [o["function"]["name"] for o in OUTILS_SCHEMA]
            console.print(Panel("\n".join(f"• [cyan]{n}[/cyan]" for n in noms), title="Outils disponibles"))
            continue

        if user_input == "/reset":
            historique.clear()
            sauvegarder_historique(historique)
            console.print("[yellow]Historique effacé.[/yellow]")
            continue

        if user_input in ("/quitter", "/q"):
            sauvegarder_historique(historique)
            console.print("[bold]À bientôt ![/bold]")
            break

        historique.append({"role": "user", "content": user_input})
        logger.info("Question : {!r}", user_input[:80])

        console.print("[bold green]Atlas :[/bold green]")
        reponse = await executer_tour(client, historique.copy())

        historique.append({"role": "assistant", "content": reponse})
        sauvegarder_historique(historique)
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
