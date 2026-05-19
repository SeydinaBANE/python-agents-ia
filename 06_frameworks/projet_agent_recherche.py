"""
MODULE 6 — Projet Final : Agent de Recherche
=============================================
Agent qui compare les frameworks du cours en combinant :
  - LangGraph  : graphe d'états pour le flux de contrôle
  - CrewAI     : équipe d'agents spécialisés
  - OpenRouter : accès unifié aux LLMs

Workflow :
  1. L'utilisateur pose une question
  2. Un agent planificateur décide : réponse directe OU recherche approfondie
  3. Si recherche → crew de 2 agents (chercheur + rédacteur)
  4. Résultat streamé avec Rich

Lance : python 06_frameworks/projet_agent_recherche.py
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()
console = Console()

MODELE_RAPIDE  = "anthropic/claude-haiku-4-5"
MODELE_QUALITE = "anthropic/claude-sonnet-4-5"


# ─── LLM ──────────────────────────────────────────────────────────────────────

def llm(modele: str = MODELE_RAPIDE) -> ChatOpenAI:
    return ChatOpenAI(
        model=modele,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )


# ─── OUTILS ───────────────────────────────────────────────────────────────────

@tool
def rechercher_web(query: str, nb_resultats: int = 3) -> str:
    """Effectue une recherche web sur un sujet. Retourne des résultats pertinents."""
    # Simulation — au Module 7 on branchera une vraie API (Tavily, Serper)
    resultats = [
        f"[Résultat {i+1}] Information sur '{query}' — source fictive"
        for i in range(nb_resultats)
    ]
    return "\n".join(resultats)


@tool
def get_meteo(ville: str) -> str:
    """Retourne la météo d'une ville."""
    meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Lyon": "15°C pluvieux"}
    return meteos.get(ville, "20°C inconnu")


@tool
def calculer(expression: str) -> str:
    """Évalue une expression mathématique."""
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return "Non autorisé"
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Erreur: {e}"


TOOLS     = [rechercher_web, get_meteo, calculer]
TOOLS_MAP = {t.name: t for t in TOOLS}


# ─── ÉTAT LANGGRAPH ───────────────────────────────────────────────────────────

class EtatRecherche(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    nb_recherches: int


# ─── NŒUDS ────────────────────────────────────────────────────────────────────

SYSTEM = """Tu es un agent de recherche expert. Pour chaque question :
- Utilise rechercher_web pour les questions factuelles ou complexes
- Utilise get_meteo uniquement pour des questions météo
- Utilise calculer pour les calculs
- Pour les questions simples sur la programmation, réponds directement
Réponds en français, de façon claire et structurée."""

async def noeud_llm(etat: EtatRecherche) -> EtatRecherche:
    model    = llm(MODELE_QUALITE)
    bound    = model.bind_tools(TOOLS)
    messages = [{"role": "system", "content": SYSTEM}] + etat["messages"]
    reponse  = await bound.ainvoke(messages)
    return {"messages": [reponse], "nb_recherches": etat["nb_recherches"]}


async def noeud_outils(etat: EtatRecherche) -> EtatRecherche:
    dernier  = etat["messages"][-1]
    resultats = []
    for tc in dernier.tool_calls:
        nom    = tc["name"]
        args   = tc["args"]
        console.print(f"  [yellow]⚙[/yellow] [cyan]{nom}[/cyan]({args})")
        outil  = TOOLS_MAP.get(nom)
        res    = outil.invoke(args) if outil else f"Tool '{nom}' inconnu"
        console.print(f"  [dim]→ {str(res)[:100]}[/dim]")
        resultats.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
    return {
        "messages":      resultats,
        "nb_recherches": etat["nb_recherches"] + 1,
    }


def router(etat: EtatRecherche) -> str:
    if etat["nb_recherches"] >= 4:
        return END
    dernier = etat["messages"][-1]
    if hasattr(dernier, "tool_calls") and dernier.tool_calls:
        return "outils"
    return END


# ─── GRAPHE ───────────────────────────────────────────────────────────────────

def construire_agent():
    g = StateGraph(EtatRecherche)
    g.add_node("llm",    noeud_llm)
    g.add_node("outils", noeud_outils)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", router, {"outils": "outils", END: END})
    g.add_edge("outils", "llm")
    return g.compile()


# ─── INTERFACE CLI ────────────────────────────────────────────────────────────

async def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print(Panel("[red]OPENROUTER_API_KEY manquante dans .env[/red]"))
        sys.exit(1)

    agent = construire_agent()

    console.print(Panel(
        "[bold green]Agent de Recherche — Module 6[/bold green]\n"
        "LangGraph + OpenRouter\n"
        "Outils : [cyan]rechercher_web[/cyan]  [cyan]get_meteo[/cyan]  [cyan]calculer[/cyan]\n"
        "Tape [cyan]/quitter[/cyan] pour sortir",
        title="Bienvenue",
    ))

    historique: list[BaseMessage] = []

    while True:
        try:
            question = Prompt.ask("\n[bold blue]Question[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]À bientôt ![/bold]")
            break

        if not question:
            continue
        if question in ("/quitter", "/q"):
            console.print("[bold]À bientôt ![/bold]")
            break

        historique.append(HumanMessage(content=question))
        etat_initial = {"messages": historique.copy(), "nb_recherches": 0}

        console.print("[bold green]Agent :[/bold green]")
        with console.status("[dim]Recherche en cours...[/dim]"):
            resultat = await agent.ainvoke(etat_initial)

        # Récupère la dernière réponse textuelle
        reponse = ""
        for msg in reversed(resultat["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                reponse = msg.content
                break

        console.print(Markdown(reponse))
        historique.append(AIMessage(content=reponse))

        nb = resultat["nb_recherches"]
        if nb > 0:
            console.print(f"[dim]{nb} recherche(s) effectuée(s)[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
