"""
MODULE 5 — Leçon 3 : Boucle ReAct (Reason + Act)
===================================================
ReAct = Reasoning + Acting.
Le LLM alterne entre :
  - Penser (Thought) : raisonner sur ce qu'il faut faire
  - Agir   (Action)  : appeler un outil
  - Observer (Observation) : lire le résultat
  jusqu'à pouvoir répondre (Final Answer).

C'est l'architecture de base de LangChain Agents, AutoGPT, etc.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()
console = Console()
MODELE  = "anthropic/claude-sonnet-4-5"


# ─── TYPES DE PAS D'EXÉCUTION ────────────────────────────────────────────────

class TypePas(str, Enum):
    THOUGHT     = "thought"      # raisonnement interne
    ACTION      = "action"       # appel d'outil
    OBSERVATION = "observation"  # résultat d'outil
    ANSWER      = "answer"       # réponse finale


@dataclass
class Pas:
    type:     TypePas
    contenu:  str
    outil:    str | None  = None
    args:     dict        = field(default_factory=dict)
    duree_ms: float       = 0.0


# ─── REGISTRE D'OUTILS ───────────────────────────────────────────────────────

# Registre inline — voir 01_tool_registry.py pour la version générique
_registry: dict[str, Any] = {}

def _tool(fn):
    _registry[fn.__name__] = fn
    return fn

@_tool
def get_meteo(ville: str) -> dict:
    """Retourne la météo d'une ville."""
    meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Lyon": "15°C pluvieux"}
    return {"ville": ville, "meteo": meteos.get(ville, "20°C inconnu")}

@_tool
def calculer(expression: str) -> dict:
    """Évalue une expression mathématique."""
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Non autorisé"}
        return {"resultat": eval(expression)}  # noqa: S307
    except Exception as e:
        return {"erreur": str(e)}

@_tool
def heure_actuelle() -> dict:
    """Retourne la date et heure actuelles."""
    from datetime import datetime
    return {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@_tool
def chercher(sujet: str, n: int = 3) -> dict:
    """Cherche des informations sur un sujet."""
    return {"sujet": sujet, "resultats": [f"Info {i+1} sur {sujet}" for i in range(n)]}

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
    {"type": "function", "function": {
        "name": "chercher", "description": "Cherche des informations",
        "parameters": {"type": "object", "properties": {
            "sujet": {"type": "string"}, "n": {"type": "integer", "default": 3},
        }, "required": ["sujet"]},
    }},
]


# ─── AGENT REACT ─────────────────────────────────────────────────────────────

SYSTEM_REACT = """Tu es un agent IA qui résout des problèmes étape par étape.

Pour chaque question :
1. Réfléchis à ce que tu dois faire (utilise les outils si nécessaire)
2. Appelle les outils pertinents
3. Synthétise les résultats en une réponse claire

Règles :
- Utilise les outils pour les faits (météo, calculs, heure) — ne devine pas
- Pour les questions générales, réponds directement sans outil
- Sois concis dans ta réponse finale"""


class AgentReAct:
    def __init__(self):
        self.client   = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.trace: list[Pas] = []

    async def executer(self, question: str, verbose: bool = True) -> str:
        """Lance la boucle ReAct et retourne la réponse finale."""
        self.trace.clear()
        messages  = [{"role": "user", "content": question}]
        iteration = 0

        if verbose:
            console.print(Rule(f"[bold blue]Question :[/bold blue] {question}"))

        while iteration < 8:
            iteration += 1
            debut = time.perf_counter()

            response = await self.client.chat.completions.create(
                model=MODELE,
                messages=[{"role": "system", "content": SYSTEM_REACT}] + messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                max_tokens=1024,
            )

            choix  = response.choices[0]
            duree  = (time.perf_counter() - debut) * 1000

            # ── Réponse finale ───────────────────────────────────────────────
            if choix.finish_reason == "stop":
                reponse = choix.message.content or ""
                self.trace.append(Pas(TypePas.ANSWER, reponse, duree_ms=duree))

                if verbose:
                    console.print(Panel(reponse, title="[bold green]Réponse finale[/bold green]"))
                return reponse

            # ── Tool calls ───────────────────────────────────────────────────
            if choix.finish_reason == "tool_calls":
                # Pensée implicite : on note les tool calls
                tool_noms = [tc.function.name for tc in choix.message.tool_calls]
                self.trace.append(Pas(TypePas.THOUGHT, f"Je vais appeler : {tool_noms}", duree_ms=duree))

                messages.append(choix.message)

                for tc in choix.message.tool_calls:
                    nom  = tc.function.name
                    args = json.loads(tc.function.arguments)

                    self.trace.append(Pas(TypePas.ACTION, f"{nom}({args})", outil=nom, args=args))

                    if verbose:
                        console.print(f"  [yellow]⚙  Action :[/yellow] [cyan]{nom}[/cyan]{args}")

                    t_debut   = time.perf_counter()
                    resultat  = _registry.get(nom, lambda **_: {"erreur": "inconnu"})(**args)
                    t_duree   = (time.perf_counter() - t_debut) * 1000

                    self.trace.append(Pas(TypePas.OBSERVATION, str(resultat), duree_ms=t_duree))

                    if verbose:
                        console.print(f"  [dim]   Obs    : {resultat}[/dim]")

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      json.dumps(resultat, ensure_ascii=False),
                    })

        return "[Erreur] Nombre max d'itérations atteint"

    def afficher_trace(self) -> None:
        """Affiche la trace d'exécution de la dernière question."""
        if not self.trace:
            console.print("[dim]Aucune trace disponible.[/dim]")
            return

        console.print(Rule("[bold]Trace d'exécution[/bold]"))
        icones = {
            TypePas.THOUGHT:     ("💭", "yellow"),
            TypePas.ACTION:      ("⚙ ", "cyan"),
            TypePas.OBSERVATION: ("👁 ", "blue"),
            TypePas.ANSWER:      ("✅", "green"),
        }
        for i, pas in enumerate(self.trace, 1):
            icone, couleur = icones[pas.type]
            console.print(
                f"  [bold {couleur}]{i}. {icone} {pas.type.value.upper()}[/bold {couleur}]"
                f"[dim] ({pas.duree_ms:.0f}ms)[/dim]"
            )
            console.print(f"     {pas.contenu[:120]}")

        total_ms = sum(p.duree_ms for p in self.trace)
        nb_tools = sum(1 for p in self.trace if p.type == TypePas.ACTION)
        console.print(f"\n  [dim]{len(self.trace)} étapes · {nb_tools} tools · {total_ms:.0f}ms total[/dim]")


# ─── DÉMONSTRATION ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 3 — Boucle ReAct[/bold]")
    agent = AgentReAct()

    questions = [
        "Quel temps fait-il à Paris et Dakar ? Laquelle est la plus chaude ?",
        "Combien font 1337 * 42 ? Et quelle heure est-il ?",
        "Qu'est-ce qu'un agent ReAct ?",   # réponse directe, sans tool
    ]

    for q in questions:
        await agent.executer(q, verbose=True)
        agent.afficher_trace()
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
