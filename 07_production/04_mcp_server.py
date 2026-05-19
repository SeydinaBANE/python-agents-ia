"""
MODULE 7 — Leçon 4 : MCP (Model Context Protocol)
====================================================
MCP est le protocole d'Anthropic pour brancher des outils externes
directement sur Claude (et d'autres LLMs compatibles).

Un serveur MCP expose des outils que Claude peut appeler nativement,
sans que tu aies à gérer la boucle agent manuellement.

Pour le tester avec Claude Desktop :
1. Lance ce serveur : python 07_production/04_mcp_server.py
2. Ajoute dans ~/Library/Application Support/Claude/claude_desktop_config.json :
   {
     "mcpServers": {
       "agents-ia": {
         "command": "python",
         "args": ["/chemin/vers/pro/07_production/04_mcp_server.py"]
       }
     }
   }
3. Relance Claude Desktop — tes outils apparaissent dans l'interface
"""

import json
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


# ─── SERVEUR MCP ─────────────────────────────────────────────────────────────

app = Server("agents-ia-tools")


# ─── DÉCLARATION DES OUTILS ───────────────────────────────────────────────────

@app.list_tools()
async def lister_outils() -> list[types.Tool]:
    """Retourne la liste des outils disponibles pour Claude."""
    return [
        types.Tool(
            name="get_meteo",
            description="Retourne la météo actuelle d'une ville (simulation).",
            inputSchema={
                "type": "object",
                "properties": {
                    "ville": {
                        "type":        "string",
                        "description": "Nom de la ville (Paris, Dakar, Lyon, Tokyo...)",
                    },
                },
                "required": ["ville"],
            },
        ),
        types.Tool(
            name="calculer",
            description="Évalue une expression mathématique de façon sécurisée.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type":        "string",
                        "description": "Expression à évaluer (ex: '15 * 8 + 3')",
                    },
                },
                "required": ["expression"],
            },
        ),
        types.Tool(
            name="heure_actuelle",
            description="Retourne la date et l'heure actuelles.",
            inputSchema={
                "type":       "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="chercher_dans_base",
            description="Cherche des informations dans la base de connaissances sur Python et les agents IA.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Question ou terme de recherche",
                    },
                    "n_resultats": {
                        "type":    "integer",
                        "default": 3,
                        "description": "Nombre de résultats à retourner",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="lire_fichier",
            description="Lit le contenu d'un fichier Python du projet de formation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chemin": {
                        "type":        "string",
                        "description": "Chemin relatif du fichier (ex: '01_fondations/01_types_variables.py')",
                    },
                },
                "required": ["chemin"],
            },
        ),
    ]


# ─── IMPLÉMENTATION DES OUTILS ───────────────────────────────────────────────

@app.call_tool()
async def executer_outil(nom: str, arguments: dict) -> list[types.TextContent]:
    """Exécute un outil et retourne son résultat à Claude."""

    if nom == "get_meteo":
        ville  = arguments["ville"]
        meteos = {
            "Paris":    {"temp": "18°C", "condition": "nuageux",     "humidite": "72%"},
            "Dakar":    {"temp": "32°C", "condition": "ensoleillé",  "humidite": "65%"},
            "Lyon":     {"temp": "15°C", "condition": "pluvieux",    "humidite": "85%"},
            "Tokyo":    {"temp": "22°C", "condition": "brumeux",     "humidite": "70%"},
            "New York": {"temp": "25°C", "condition": "ensoleillé",  "humidite": "55%"},
        }
        info    = meteos.get(ville, {"temp": "20°C", "condition": "inconnu", "humidite": "N/A"})
        resultat = f"Météo à {ville} : {info['temp']}, {info['condition']}, humidité {info['humidite']}"
        return [types.TextContent(type="text", text=resultat)]

    if nom == "calculer":
        expression = arguments["expression"]
        try:
            autorise = set("0123456789+-*/()., ")
            if not all(c in autorise for c in expression):
                return [types.TextContent(type="text", text="Erreur : expression non autorisée")]
            resultat = eval(expression)  # noqa: S307
            return [types.TextContent(type="text", text=f"{expression} = {resultat}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Erreur de calcul : {e}")]

    if nom == "heure_actuelle":
        maintenant = datetime.now().strftime("%A %d %B %Y à %H:%M:%S")
        return [types.TextContent(type="text", text=f"Date et heure : {maintenant}")]

    if nom == "chercher_dans_base":
        query      = arguments["query"]
        n          = arguments.get("n_resultats", 3)
        resultats  = [
            f"Résultat {i+1} sur '{query}' : information de démonstration #{i+1}"
            for i in range(n)
        ]
        texte = f"Recherche pour '{query}' :\n" + "\n".join(f"• {r}" for r in resultats)
        return [types.TextContent(type="text", text=texte)]

    if nom == "lire_fichier":
        from pathlib import Path
        chemin = Path(arguments["chemin"])
        if not chemin.exists():
            return [types.TextContent(type="text", text=f"Fichier introuvable : {chemin}")]
        if chemin.suffix != ".py":
            return [types.TextContent(type="text", text="Seuls les fichiers .py sont autorisés")]
        contenu = chemin.read_text(encoding="utf-8")
        return [types.TextContent(type="text", text=contenu[:3000])]   # limite 3000 chars

    return [types.TextContent(type="text", text=f"Outil inconnu : {nom}")]


# ─── RESSOURCES MCP (optionnel) ───────────────────────────────────────────────
# Les ressources exposent des données statiques que Claude peut lire

@app.list_resources()
async def lister_ressources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="formation://modules",
            name="Plan de formation",
            description="Liste des 7 modules de la formation Python → Agents IA",
            mimeType="text/plain",
        ),
    ]

@app.read_resource()
async def lire_ressource(uri: str) -> str:
    if uri == "formation://modules":
        return "\n".join([
            "Module 1 : Python Fondations",
            "Module 2 : Python Intermédiaire (async, POO)",
            "Module 3 : Écosystème IA (pydantic, httpx, loguru)",
            "Module 4 : LLM APIs (tool use, streaming)",
            "Module 5 : Architecture Agent (ReAct, mémoire)",
            "Module 6 : Frameworks (LangChain, LangGraph, CrewAI)",
            "Module 7 : Production (FastAPI, MCP, RAG, Docker)",
        ])
    return "Ressource introuvable"


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
