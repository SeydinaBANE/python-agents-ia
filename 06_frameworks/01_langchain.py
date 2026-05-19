"""
MODULE 6 — Leçon 1 : LangChain
================================
LangChain est le framework le plus populaire pour les agents IA.
Il standardise les composants : LLMs, tools, prompts, mémoire, chaînes.

Ce fichier couvre les concepts fondamentaux avec OpenRouter.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# LangChain utilise langchain-openai pour se connecter à tout provider OpenAI-compatible
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor


# ─── 1. LLM VIA OPENROUTER ────────────────────────────────────────────────────

def get_llm(modele: str = "anthropic/claude-haiku-4-5", temperature: float = 0.3) -> ChatOpenAI:
    """Crée un LLM LangChain qui pointe vers OpenRouter."""
    return ChatOpenAI(
        model=modele,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )


# ─── 2. APPELS DIRECTS ────────────────────────────────────────────────────────

async def demo_appels_directs():
    llm = get_llm()

    # Message simple
    response = await llm.ainvoke("Qu'est-ce que LangChain en une phrase ?")
    console.print(f"[green]Réponse :[/green] {response.content}")

    # Avec system prompt
    messages = [
        SystemMessage(content="Tu es un expert Python. Réponds en 2 phrases max."),
        HumanMessage(content="Pourquoi utiliser LangChain ?"),
    ]
    response = await llm.ainvoke(messages)
    console.print(f"[green]Avec system :[/green] {response.content}")

    # Streaming
    console.print("\n[bold]Streaming :[/bold]")
    print("Bot : ", end="", flush=True)
    async for chunk in llm.astream("Explique les agents IA en 3 points."):
        print(chunk.content, end="", flush=True)
    print()


# ─── 3. LCEL — LangChain Expression Language ──────────────────────────────────
# LCEL permet d'enchaîner des composants avec l'opérateur |
# prompt | llm | parser  →  pipeline complet

async def demo_lcel():
    llm    = get_llm()
    parser = StrOutputParser()

    # Chaîne simple : prompt → LLM → string
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un expert en {domaine}. Réponds en {langue}, maximum {mots} mots."),
        ("human",  "{question}"),
    ])

    chaine = prompt | llm | parser

    reponse = await chaine.ainvoke({
        "domaine":  "agents IA",
        "langue":   "français",
        "mots":     "50",
        "question": "Qu'est-ce qu'une boucle ReAct ?",
    })
    console.print(f"\n[cyan]LCEL :[/cyan] {reponse}")

    # Batch : plusieurs questions en parallèle
    questions = [
        {"domaine": "Python",  "langue": "français", "mots": "30", "question": "Qu'est-ce qu'un générateur ?"},
        {"domaine": "LLM",     "langue": "français", "mots": "30", "question": "Qu'est-ce que le fine-tuning ?"},
        {"domaine": "agents",  "langue": "français", "mots": "30", "question": "Qu'est-ce qu'un tool ?"},
    ]
    reponses = await chaine.abatch(questions)
    for q, r in zip(questions, reponses):
        console.print(f"  [yellow]Q:[/yellow] {q['question']}")
        console.print(f"  [green]R:[/green] {r}\n")


# ─── 4. TOOLS LANGCHAIN ───────────────────────────────────────────────────────
# Le décorateur @tool génère automatiquement le schéma depuis la docstring

@tool
def get_meteo(ville: str) -> str:
    """Retourne la météo actuelle d'une ville.

    Args:
        ville: Nom de la ville (ex: Paris, Dakar, Lyon)
    """
    meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Lyon": "15°C pluvieux"}
    return meteos.get(ville, f"Météo inconnue pour {ville}")


@tool
def calculer(expression: str) -> str:
    """Évalue une expression mathématique.

    Args:
        expression: Expression à évaluer (ex: '15 * 8 + 3')
    """
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return "Expression non autorisée"
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Erreur : {e}"


@tool
def chercher_info(sujet: str) -> str:
    """Cherche des informations sur un sujet.

    Args:
        sujet: Sujet à rechercher
    """
    return f"Résultats de recherche pour '{sujet}': [info 1, info 2, info 3]"


# ─── 5. AGENT LANGCHAIN ───────────────────────────────────────────────────────

async def demo_agent():
    llm   = get_llm(modele="anthropic/claude-sonnet-4-5")
    tools = [get_meteo, calculer, chercher_info]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant IA utile. Réponds en français."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent          = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    questions = [
        "Quel temps fait-il à Dakar ?",
        "Combien font 144 * 25 ?",
        "Quel temps à Paris et Lyon ? Laquelle est la plus chaude ?",
    ]

    for q in questions:
        console.print(f"\n[bold blue]Question :[/bold blue] {q}")
        result = await agent_executor.ainvoke({"input": q})
        console.print(f"[bold green]Réponse :[/bold green] {result['output']}")


# ─── 6. MÉMOIRE LANGCHAIN ─────────────────────────────────────────────────────

async def demo_memoire():
    """Mémoire conversationnelle manuelle avec LCEL."""
    llm      = get_llm()
    parser   = StrOutputParser()
    historique: list = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant IA. Réponds en français."),
        MessagesPlaceholder("historique"),
        ("human", "{input}"),
    ])
    chaine = prompt | llm | parser

    echanges = [
        "Bonjour ! Je m'appelle Seydina et j'apprends Python.",
        "Quel est mon prénom ?",
        "Que suis-je en train d'apprendre ?",
    ]

    for msg in echanges:
        console.print(f"\n[blue]Toi :[/blue] {msg}")
        reponse = await chaine.ainvoke({"input": msg, "historique": historique})
        historique.extend([HumanMessage(content=msg), SystemMessage(content=reponse)])
        console.print(f"[green]Bot :[/green] {reponse}")


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 1 — LangChain[/bold]")

    console.print("\n[bold cyan]1. Appels directs[/bold cyan]")
    await demo_appels_directs()

    console.print("\n[bold cyan]2. LCEL (pipelines)[/bold cyan]")
    await demo_lcel()

    console.print("\n[bold cyan]3. Agent avec tools[/bold cyan]")
    await demo_agent()

    console.print("\n[bold cyan]4. Mémoire conversationnelle[/bold cyan]")
    await demo_memoire()


if __name__ == "__main__":
    asyncio.run(main())
