"""
MODULE 4 — Leçon 3 : Tool Use (Function Calling)  ← LA leçon clé des agents
==============================================================================
Le tool use permet au LLM de décider d'appeler une fonction Python.
C'est le mécanisme fondamental des agents : le LLM "pense", choisit un outil,
l'agent l'exécute, puis renvoie le résultat au LLM pour qu'il continue.

Flux complet :
  1. Tu envoies la question + la liste des tools disponibles
  2. Le LLM répond avec un "tool_call" (nom + arguments JSON)
  3. Tu exécutes la fonction Python correspondante
  4. Tu renvoies le résultat au LLM
  5. Le LLM génère la réponse finale pour l'utilisateur
"""

import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
MODELE = "anthropic/claude-sonnet-4-5"   # sonnet gère mieux le tool use

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


# ─── 1. DÉFINIR LES OUTILS ────────────────────────────────────────────────────
# Le format est celui d'OpenAI — un JSON Schema pour chaque outil.
# Pydantic peut le générer automatiquement (on le verra au Module 5).

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "get_meteo",
            "description": "Retourne la météo actuelle d'une ville.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ville": {
                        "type": "string",
                        "description": "Nom de la ville (ex: Paris, Dakar, Lyon)",
                    },
                    "unite": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unité de température",
                    },
                },
                "required": ["ville"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculer",
            "description": "Effectue un calcul mathématique simple.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Expression mathématique (ex: '15 * 8 + 3')",
                    },
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
                    "sujet": {"type": "string", "description": "Sujet à rechercher"},
                    "nb_resultats": {"type": "integer", "default": 3},
                },
                "required": ["sujet"],
            },
        },
    },
]


# ─── 2. IMPLÉMENTER LES OUTILS (fonctions Python réelles) ────────────────────

def get_meteo(ville: str, unite: str = "celsius") -> dict:
    """Simulation d'une API météo."""
    donnees = {
        "Paris":  {"temp": 18, "condition": "nuageux"},
        "Dakar":  {"temp": 32, "condition": "ensoleillé"},
        "Lyon":   {"temp": 15, "condition": "pluvieux"},
        "Tokyo":  {"temp": 22, "condition": "brumeux"},
    }
    info = donnees.get(ville, {"temp": 20, "condition": "inconnu"})
    temp = info["temp"]
    if unite == "fahrenheit":
        temp = round(temp * 9/5 + 32, 1)
    return {
        "ville":      ville,
        "temperature": f"{temp}°{'F' if unite == 'fahrenheit' else 'C'}",
        "condition":  info["condition"],
    }

def calculer(expression: str) -> dict:
    """Évalue une expression mathématique de façon sécurisée."""
    try:
        # eval limité aux opérations arithmétiques simples
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Expression non autorisée"}
        resultat = eval(expression)  # noqa: S307
        return {"expression": expression, "resultat": resultat}
    except Exception as e:
        return {"erreur": str(e)}

def chercher_info(sujet: str, nb_resultats: int = 3) -> dict:
    """Simulation d'une recherche web."""
    resultats_faux = [
        f"[Résultat {i+1}] Information sur '{sujet}' — source fictive {i+1}"
        for i in range(nb_resultats)
    ]
    return {"sujet": sujet, "resultats": resultats_faux}


# Registre : nom → fonction Python
REGISTRE: dict[str, callable] = {
    "get_meteo":    get_meteo,
    "calculer":     calculer,
    "chercher_info": chercher_info,
}


# ─── 3. BOUCLE AGENT — le cœur du mécanisme ──────────────────────────────────

async def agent_avec_tools(question: str, verbose: bool = True) -> str:
    """
    Boucle agent complète :
    1. Envoie la question avec les tools disponibles
    2. Si le LLM appelle un tool → l'exécute → renvoie le résultat
    3. Répète jusqu'à obtenir une réponse finale (stop_reason = "stop")
    """
    client   = get_client()
    messages = [{"role": "user", "content": question}]

    if verbose:
        console.print(f"\n[bold blue]Question :[/bold blue] {question}")

    iteration = 0
    while True:
        iteration += 1
        if verbose:
            console.print(f"[dim]→ Appel LLM #{iteration}[/dim]")

        response = await client.chat.completions.create(
            model=MODELE,
            messages=messages,
            tools=OUTILS,
            tool_choice="auto",   # le LLM choisit si/quand utiliser un tool
            max_tokens=1024,
        )

        choix = response.choices[0]

        # ── Cas 1 : le LLM a fini, réponse finale ──────────────────────────
        if choix.finish_reason == "stop":
            reponse_finale = choix.message.content or ""
            if verbose:
                console.print(Panel(reponse_finale, title="[bold green]Réponse finale[/bold green]"))
            return reponse_finale

        # ── Cas 2 : le LLM demande d'appeler un ou plusieurs tools ─────────
        if choix.finish_reason == "tool_calls":
            # Ajouter le message du LLM (avec ses tool_calls) à l'historique
            messages.append(choix.message)

            # Exécuter chaque tool demandé
            for tool_call in choix.message.tool_calls:
                nom    = tool_call.function.name
                args   = json.loads(tool_call.function.arguments)

                if verbose:
                    console.print(f"[yellow]  ⚙ Tool appelé :[/yellow] [cyan]{nom}[/cyan]({args})")

                # Exécuter la fonction Python
                if nom in REGISTRE:
                    resultat = REGISTRE[nom](**args)
                else:
                    resultat = {"erreur": f"Tool '{nom}' inconnu"}

                if verbose:
                    console.print(f"[dim]    Résultat : {resultat}[/dim]")

                # Ajouter le résultat du tool à l'historique
                # Le rôle "tool" est obligatoire — le LLM attend ce format
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      json.dumps(resultat, ensure_ascii=False),
                })

        # Sécurité : ne pas boucler indéfiniment
        if iteration >= 10:
            return "[Erreur] Trop d'itérations"


# ─── 4. TOOL CHOICE — contrôler quand utiliser les outils ─────────────────────

async def demo_tool_choice():
    client = get_client()

    # "auto"     → le LLM choisit (par défaut)
    # "required" → le LLM DOIT appeler un tool
    # "none"     → le LLM ne peut PAS appeler de tool
    # {"type": "function", "function": {"name": "get_meteo"}} → force un tool précis

    response = await client.chat.completions.create(
        model=MODELE,
        messages=[{"role": "user", "content": "Bonjour, comment vas-tu ?"}],
        tools=OUTILS,
        tool_choice="none",     # pas de tool pour une question simple
        max_tokens=100,
    )
    console.print(f"[dim]Sans tool : {response.choices[0].message.content}[/dim]")


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 3 — Tool Use[/bold]")

    # Test 1 : question qui nécessite un seul tool
    await agent_avec_tools("Quel temps fait-il à Dakar ?")

    # Test 2 : question qui nécessite plusieurs tools en séquence
    await agent_avec_tools("Quel temps fait-il à Paris et à Tokyo ? Compare les deux villes.")

    # Test 3 : question qui combine calcul et recherche
    await agent_avec_tools("Combien font 847 * 23 ? Et cherche des infos sur les agents IA.")

    # Test 4 : question sans besoin de tool
    await agent_avec_tools("Qu'est-ce qu'un LLM ?")

    # Test 5 : tool_choice
    console.print("\n[bold cyan]5. tool_choice='none'[/bold cyan]")
    await demo_tool_choice()


if __name__ == "__main__":
    asyncio.run(main())
