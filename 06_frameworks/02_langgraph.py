"""
MODULE 6 — Leçon 2 : LangGraph
================================
LangGraph modélise un agent comme un graphe d'états.
Chaque nœud est une fonction, chaque arc est une transition.
C'est la bonne approche pour les agents complexes avec des branches,
des boucles, et plusieurs chemins d'exécution possibles.

Exemple : agent qui peut soit appeler un outil, soit répondre directement,
avec une boucle de retry si le résultat est insuffisant.
"""

import asyncio
import json
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


# ─── 1. ÉTAT DU GRAPHE ────────────────────────────────────────────────────────
# L'état est le seul objet qui circule entre les nœuds.
# add_messages est un "reducer" : il ajoute les messages au lieu de les remplacer.

class EtatAgent(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─── 2. OUTILS ───────────────────────────────────────────────────────────────

@tool
def get_meteo(ville: str) -> str:
    """Météo d'une ville."""
    meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Lyon": "15°C pluvieux"}
    return meteos.get(ville, "20°C inconnu")

@tool
def calculer(expression: str) -> str:
    """Calcule une expression mathématique."""
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return "Non autorisé"
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Erreur: {e}"

TOOLS = [get_meteo, calculer]
TOOLS_MAP = {t.name: t for t in TOOLS}


# ─── 3. NŒUDS DU GRAPHE ──────────────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(
        model="anthropic/claude-sonnet-4-5",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    ).bind_tools(TOOLS)


async def noeud_llm(etat: EtatAgent) -> EtatAgent:
    """Nœud LLM : appelle le modèle et retourne sa réponse."""
    llm      = get_llm()
    messages = etat["messages"]
    reponse  = await llm.ainvoke(messages)
    return {"messages": [reponse]}


async def noeud_outils(etat: EtatAgent) -> EtatAgent:
    """Nœud Outils : exécute tous les tool_calls du dernier message."""
    dernier_msg = etat["messages"][-1]
    resultats   = []

    for tc in dernier_msg.tool_calls:
        nom     = tc["name"]
        args    = tc["args"]
        console.print(f"  [yellow]⚙[/yellow] [cyan]{nom}[/cyan]({args})")

        outil   = TOOLS_MAP.get(nom)
        resultat = outil.invoke(args) if outil else f"Outil '{nom}' inconnu"

        console.print(f"  [dim]→ {resultat}[/dim]")
        resultats.append(ToolMessage(content=str(resultat), tool_call_id=tc["id"]))

    return {"messages": resultats}


# ─── 4. CONDITION DE ROUTAGE ──────────────────────────────────────────────────
# Cette fonction décide vers quel nœud aller après le LLM

def router(etat: EtatAgent) -> str:
    """
    Si le LLM a demandé des tools → aller au nœud outils.
    Sinon → fin.
    """
    dernier_msg = etat["messages"][-1]
    if hasattr(dernier_msg, "tool_calls") and dernier_msg.tool_calls:
        return "outils"
    return END


# ─── 5. CONSTRUIRE LE GRAPHE ──────────────────────────────────────────────────

def construire_graphe() -> StateGraph:
    graphe = StateGraph(EtatAgent)

    # Ajouter les nœuds
    graphe.add_node("llm",    noeud_llm)
    graphe.add_node("outils", noeud_outils)

    # Point d'entrée
    graphe.set_entry_point("llm")

    # Arcs conditionnels depuis llm
    graphe.add_conditional_edges(
        source="llm",
        path=router,
        path_map={"outils": "outils", END: END},
    )

    # Après les outils → retour au LLM (boucle)
    graphe.add_edge("outils", "llm")

    return graphe.compile()


# ─── 6. GRAPHE AVEC ÉTAT ÉTENDU ──────────────────────────────────────────────
# Exemple plus avancé : état avec compteur d'itérations et validation

class EtatAvance(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    iterations:  int
    question:    str
    satisfait:   bool


async def noeud_llm_avance(etat: EtatAvance) -> EtatAvance:
    llm      = get_llm()
    reponse  = await llm.ainvoke(etat["messages"])
    return {
        "messages":   [reponse],
        "iterations": etat["iterations"] + 1,
    }

async def noeud_outils_avance(etat: EtatAvance) -> EtatAvance:
    dernier_msg = etat["messages"][-1]
    resultats   = []
    for tc in dernier_msg.tool_calls:
        outil    = TOOLS_MAP.get(tc["name"])
        resultat = outil.invoke(tc["args"]) if outil else "inconnu"
        resultats.append(ToolMessage(content=str(resultat), tool_call_id=tc["id"]))
    return {"messages": resultats}

def router_avance(etat: EtatAvance) -> str:
    if etat["iterations"] >= 5:
        return END
    dernier_msg = etat["messages"][-1]
    if hasattr(dernier_msg, "tool_calls") and dernier_msg.tool_calls:
        return "outils"
    return END


# ─── DÉMONSTRATION ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 2 — LangGraph[/bold]")

    agent = construire_graphe()

    questions = [
        "Quel temps fait-il à Dakar ?",
        "Quel temps à Paris et Lyon ? Calcule la somme des températures (18 + 15).",
        "Qu'est-ce qu'un agent IA ?",   # sans tool
    ]

    for q in questions:
        console.print(f"\n[bold blue]Question :[/bold blue] {q}")
        etat_initial = {"messages": [HumanMessage(content=q)]}

        resultat = await agent.ainvoke(etat_initial)
        reponse_finale = resultat["messages"][-1].content
        console.print(Panel(reponse_finale, title="[bold green]Réponse[/bold green]"))


if __name__ == "__main__":
    asyncio.run(main())
