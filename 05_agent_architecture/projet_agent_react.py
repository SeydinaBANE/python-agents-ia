"""
MODULE 5 — Projet Final : Agent ReAct avec mémoire et registre
================================================================
Cet agent assemble TOUTES les briques du Module 5 :
  - Tool registry automatique    (leçon 1)
  - Mémoire courte + longue      (leçon 2)
  - Boucle ReAct complète        (leçon 3)
  - Planification multi-étapes   (leçon 4)

Lance : python 05_agent_architecture/projet_agent_react.py
Commandes : /memoire  /trace  /plan <question>  /reset  /quitter
"""

import asyncio
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

load_dotenv()
console = Console()
MODELE  = "anthropic/claude-sonnet-4-5"
Path("05_agent_architecture/data").mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add("05_agent_architecture/data/agent.log", level="DEBUG", rotation="1 MB")


# ─── TOOL REGISTRY ───────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, description: str):
        def dec(fn: Callable) -> Callable:
            sig    = inspect.signature(fn)
            props  = {}
            req    = []
            for nom, p in sig.parameters.items():
                ann = p.annotation
                t   = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(ann, "string")
                props[nom] = {"type": t}
                if p.default is inspect.Parameter.empty:
                    req.append(nom)
                else:
                    props[nom]["default"] = p.default
            self._tools[fn.__name__] = {
                "fn": fn,
                "schema": {"type": "function", "function": {
                    "name": fn.__name__, "description": description,
                    "parameters": {"type": "object", "properties": props, "required": req},
                }},
            }
            return fn
        return dec

    def run(self, nom: str, **kwargs) -> Any:
        if nom not in self._tools:
            return {"erreur": f"Tool '{nom}' inconnu"}
        try:
            return self._tools[nom]["fn"](**kwargs)
        except Exception as e:
            return {"erreur": str(e)}

    def schemas(self) -> list[dict]:
        return [t["schema"] for t in self._tools.values()]

    def noms(self) -> list[str]:
        return list(self._tools.keys())


# ─── OUTILS ───────────────────────────────────────────────────────────────────

registry = ToolRegistry()

@registry.register("Météo actuelle d'une ville")
def get_meteo(ville: str) -> dict:
    meteos = {"Paris": ("18°C", "nuageux"), "Dakar": ("32°C", "ensoleillé"),
              "Lyon": ("15°C", "pluvieux"), "Tokyo": ("22°C", "brumeux")}
    t, c = meteos.get(ville, ("20°C", "inconnu"))
    return {"ville": ville, "temperature": t, "condition": c}

@registry.register("Calcule une expression mathématique")
def calculer(expression: str) -> dict:
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Non autorisé"}
        return {"resultat": eval(expression)}  # noqa: S307
    except Exception as e:
        return {"erreur": str(e)}

@registry.register("Cherche des informations sur un sujet")
def chercher(sujet: str, n: int = 3) -> dict:
    return {"sujet": sujet, "resultats": [f"Info {i+1} sur '{sujet}'" for i in range(n)]}

@registry.register("Retourne la date et l'heure actuelles")
def heure_actuelle() -> dict:
    return {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@registry.register("Mémorise un fait important sur l'utilisateur")
def memoriser_fait(fait: str, categorie: str = "general") -> dict:
    return {"statut": "mémorisé", "fait": fait, "categorie": categorie}


# ─── MÉMOIRE ──────────────────────────────────────────────────────────────────

class Memoire:
    def __init__(self):
        self.messages:  list[dict] = []
        self.souvenirs: list[dict] = []
        self._chemin = Path("05_agent_architecture/data/memoire.json")
        self._charger()

    def _charger(self):
        if self._chemin.exists():
            try:
                self.souvenirs = json.loads(self._chemin.read_text(encoding="utf-8"))
            except Exception:
                self.souvenirs = []

    def _sauvegarder(self):
        self._chemin.write_text(
            json.dumps(self.souvenirs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def ajouter(self, role: str, content: Any):
        self.messages.append({"role": role, "content": content})
        # Limite la mémoire courte à 30 messages
        if len(self.messages) > 30:
            self.messages = self.messages[:1] + self.messages[-29:]

    def memoriser(self, fait: str, categorie: str = "general"):
        self.souvenirs.append({
            "fait": fait, "categorie": categorie,
            "date": datetime.now().isoformat()[:19],
        })
        self._sauvegarder()

    def contexte(self, question: str = "") -> str:
        if not self.souvenirs:
            return ""
        q = question.lower()
        pertinents = [s for s in self.souvenirs if not q or q[:20] in s["fait"].lower()][:5]
        if not pertinents:
            pertinents = self.souvenirs[-5:]
        return "## Mémoire :\n" + "\n".join(f"- [{s['categorie']}] {s['fait']}" for s in pertinents)

    def pour_api(self, question: str = "") -> list[dict]:
        system = SYSTEM_PROMPT
        ctx    = self.contexte(question)
        if ctx:
            system += "\n\n" + ctx
        return [{"role": "system", "content": system}] + self.messages

    def effacer_conversation(self):
        self.messages.clear()


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es Atlas, un agent IA expert en Python et développement d'agents.

## Processus
1. Analyse la question
2. Utilise les outils si nécessaire (météo, calculs, recherche, heure)
3. Si tu apprends un fait important sur l'utilisateur, utilise memoriser_fait
4. Synthétise une réponse claire

## Règles
- Réponds en français, sois concis
- Pour les calculs, utilise TOUJOURS l'outil calculer
- Ne devine pas les faits (météo, heure) — utilise les outils"""


# ─── BOUCLE AGENT ────────────────────────────────────────────────────────────

class AgentReAct:
    def __init__(self):
        self.client   = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.memoire  = Memoire()
        self.last_trace: list[dict] = []

    async def repondre(self, question: str) -> str:
        self.last_trace = []
        messages = self.memoire.pour_api(question)
        iteration = 0

        while iteration < 8:
            iteration += 1
            response = await self.client.chat.completions.create(
                model=MODELE,
                messages=messages,
                tools=registry.schemas(),
                tool_choice="auto",
                max_tokens=1024,
            )
            choix = response.choices[0]

            # ── Réponse finale — streamée ────────────────────────────────────
            if choix.finish_reason == "stop":
                reponse_finale = ""
                with Live(console=console, refresh_per_second=15) as live:
                    async with await self.client.chat.completions.create(
                        model=MODELE,
                        messages=messages,
                        max_tokens=1024,
                        stream=True,
                    ) as stream:
                        async for chunk in stream:
                            tok = chunk.choices[0].delta.content or ""
                            reponse_finale += tok
                            live.update(Markdown(reponse_finale))
                return reponse_finale

            # ── Tool calls ───────────────────────────────────────────────────
            if choix.finish_reason == "tool_calls":
                messages.append(choix.message)

                for tc in choix.message.tool_calls:
                    nom  = tc.function.name
                    args = json.loads(tc.function.arguments)

                    console.print(f"  [yellow]⚙[/yellow] [cyan]{nom}[/cyan] {args}")
                    logger.debug("Tool: {}({})", nom, args)

                    debut    = time.perf_counter()
                    resultat = registry.run(nom, **args)
                    duree_ms = (time.perf_counter() - debut) * 1000

                    self.last_trace.append({"tool": nom, "args": args, "result": resultat, "ms": round(duree_ms)})

                    # Intercepte memoriser_fait pour persister
                    if nom == "memoriser_fait":
                        self.memoire.memoriser(args.get("fait", ""), args.get("categorie", "general"))

                    console.print(f"  [dim]→ {str(resultat)[:100]}[/dim]")

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      json.dumps(resultat, ensure_ascii=False),
                    })

        return "[Erreur] Trop d'itérations"


# ─── INTERFACE CLI ───────────────────────────────────────────────────────────

async def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print(Panel("[red]OPENROUTER_API_KEY manquante dans .env[/red]"))
        sys.exit(1)

    agent = AgentReAct()

    console.print(Panel(
        "[bold green]Agent Atlas — Module 5[/bold green]\n"
        f"Modèle : [cyan]{MODELE}[/cyan] · Outils : [cyan]{', '.join(registry.noms())}[/cyan]\n"
        "Commandes : [cyan]/memoire[/cyan]  [cyan]/trace[/cyan]  [cyan]/reset[/cyan]  [cyan]/quitter[/cyan]",
        title="ReAct Agent",
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]Toi[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]À bientôt ![/bold]")
            break

        if not user_input:
            continue

        if user_input == "/memoire":
            if agent.memoire.souvenirs:
                for s in agent.memoire.souvenirs:
                    console.print(f"  [cyan][{s['categorie']}][/cyan] {s['fait']}")
            else:
                console.print("[dim]Aucun souvenir enregistré.[/dim]")
            continue

        if user_input == "/trace":
            if agent.last_trace:
                for t in agent.last_trace:
                    console.print(f"  [yellow]⚙[/yellow] {t['tool']}({t['args']}) → {str(t['result'])[:80]} [dim]({t['ms']}ms)[/dim]")
            else:
                console.print("[dim]Aucune trace disponible.[/dim]")
            continue

        if user_input == "/reset":
            agent.memoire.effacer_conversation()
            console.print("[yellow]Conversation effacée (mémoire longue conservée).[/yellow]")
            continue

        if user_input in ("/quitter", "/q"):
            console.print("[bold]À bientôt ![/bold]")
            break

        agent.memoire.ajouter("user", user_input)
        logger.info("Question : {!r}", user_input[:80])

        console.print("[bold green]Atlas :[/bold green]")
        reponse = await agent.repondre(user_input)
        agent.memoire.ajouter("assistant", reponse)
        logger.info("Réponse générée ({} chars)", len(reponse))


if __name__ == "__main__":
    asyncio.run(main())
