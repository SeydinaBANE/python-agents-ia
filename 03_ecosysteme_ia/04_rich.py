"""
MODULE 3 — Leçon 4 : Rich
===========================
Rich transforme ton terminal en interface soignée.
Indispensable pour les agents CLI : progress bars pendant les appels LLM,
tableaux de résultats, affichage coloré des réponses.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.columns import Columns
from rich.text import Text
from rich import print as rprint
import time

console = Console()


# ─── PRINT ENRICHI ────────────────────────────────────────────────────────────

rprint("[bold green]Succès ![/bold green]")
rprint("[bold red]Erreur critique[/bold red]")
rprint("[cyan]Info[/cyan] : agent démarré")

# Styles : bold, italic, underline, strike
# Couleurs : red, green, blue, yellow, cyan, magenta, white, black
# Fond    : on_red, on_green, on_blue...


# ─── PANEL — encadrer un bloc ─────────────────────────────────────────────────

console.print(Panel(
    "[bold]Agent Atlas[/bold]\nModèle : claude-haiku-4-5\nTokens utilisés : 1 250 / 10 000",
    title="[cyan]Statut de l'agent[/cyan]",
    border_style="green",
))

console.print(Panel(
    "[red]Clé API manquante — configure .env[/red]",
    title="Erreur",
    border_style="red",
))


# ─── TABLE ────────────────────────────────────────────────────────────────────

def afficher_historique(historique: list[dict]) -> None:
    table = Table(title="Historique de conversation", show_lines=True)
    table.add_column("#",        style="dim",   width=4)
    table.add_column("Rôle",     style="cyan",  width=12)
    table.add_column("Message",  style="white", min_width=40)
    table.add_column("Tokens",   style="yellow", width=8)

    for i, msg in enumerate(historique, 1):
        role    = msg["role"]
        contenu = msg["content"]
        tokens  = str(len(contenu) // 4)
        extrait = contenu[:80] + "…" if len(contenu) > 80 else contenu

        couleur_role = "bold green" if role == "assistant" else "bold blue"
        table.add_row(str(i), f"[{couleur_role}]{role}[/{couleur_role}]", extrait, tokens)

    console.print(table)

historique_demo = [
    {"role": "user",      "content": "Qu'est-ce qu'un agent IA ?"},
    {"role": "assistant", "content": "Un agent IA est un système autonome qui perçoit son environnement, raisonne, et agit pour atteindre un objectif."},
    {"role": "user",      "content": "Donne un exemple."},
    {"role": "assistant", "content": "ChatGPT avec des plugins est un agent : il peut chercher sur le web, exécuter du code, et appeler des APIs externes."},
]
afficher_historique(historique_demo)


# ─── MARKDOWN — afficher les réponses LLM formatées ──────────────────────────
# Les LLMs retournent souvent du Markdown — rich le rend correctement

reponse_llm = """
## Résultat de l'analyse

L'agent a identifié **3 actions** à effectuer :

1. Rechercher les données météo
2. Calculer la moyenne des températures
3. Générer un rapport

```python
def analyser(donnees: list[float]) -> dict:
    return {"moyenne": sum(donnees) / len(donnees)}
```

> **Note** : L'agent a utilisé 450 tokens pour cette analyse.
"""

console.print(Markdown(reponse_llm))


# ─── SYNTAX HIGHLIGHTING ─────────────────────────────────────────────────────

code = '''
async def agent_loop(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    while True:
        response = await llm(messages)
        if response.stop_reason == "end_turn":
            return response.content
        tool_result = await execute_tool(response.tool_call)
        messages.append(tool_result)
'''

console.print(Syntax(code, "python", theme="monokai", line_numbers=True))


# ─── PROGRESS — spinner pendant les appels LLM ───────────────────────────────

def simuler_appel_avec_spinner(label: str, duree: float = 0.5) -> str:
    with console.status(f"[dim]{label}...[/dim]", spinner="dots"):
        time.sleep(duree)
    return f"Résultat de : {label}"

simuler_appel_avec_spinner("Interrogation du LLM")
simuler_appel_avec_spinner("Exécution du tool météo")


# ─── PROGRESS BAR — traitement par lots ──────────────────────────────────────

def traiter_documents(documents: list[str]) -> list[str]:
    resultats = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        tache = progress.add_task("Traitement des documents", total=len(documents))
        for doc in documents:
            time.sleep(0.1)     # simule le traitement
            resultats.append(doc.upper())
            progress.advance(tache)
    return resultats

docs = [f"document_{i}.txt" for i in range(8)]
traiter_documents(docs)


# ─── LIVE — mise à jour dynamique ─────────────────────────────────────────────
# Utile pour afficher l'état d'un agent qui s'exécute

def afficher_etat_agent():
    etapes = [
        "Analyse de la question...",
        "Recherche d'informations...",
        "Appel tool get_meteo(Paris)...",
        "Synthèse de la réponse...",
        "Réponse prête !",
    ]
    with Live(console=console, refresh_per_second=4) as live:
        for etape in etapes:
            live.update(Panel(f"[cyan]⚙ {etape}[/cyan]", title="Agent en cours"))
            time.sleep(0.4)

afficher_etat_agent()


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée une fonction 'afficher_comparaison_modeles(resultats)' qui affiche
   un tableau rich avec colonnes : Modèle, Durée, Tokens, Extrait réponse.
   Colorie en vert le modèle le plus rapide, en rouge le plus lent.

2. Crée une fonction 'afficher_tools(registre)' qui liste les outils
   disponibles dans un Panel avec leur nom et description.

3. Utilise Rich Live pour simuler le streaming : affiche chaque mot
   d'une phrase avec un délai de 0.1s, comme si c'était un vrai stream LLM.
"""
