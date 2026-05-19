"""
MODULE 1 — Projet Final : Chatbot CLI avec historique
======================================================
Ce projet réunit TOUTES les leçons du module :
- Types & variables       (leçon 1)
- Contrôle de flux        (leçon 2)
- Fonctions               (leçon 3)
- Gestion d'erreurs       (leçon 4)
- Fichiers & JSON         (leçon 5)
- Modules & packages      (leçon 6)

Lance : python 01_fondations/projet_chatbot.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FICHIER_HISTORIQUE = Path("01_fondations/data/chatbot_historique.json")
MODELE = "anthropic/claude-haiku-4-5"      # haiku = rapide et économique
SYSTEM_PROMPT = "Tu es un assistant pédagogique qui enseigne Python et les agents IA. Réponds en français, de manière claire et concise."

console = Console()


# ─── GESTION DE L'HISTORIQUE (leçons 4 + 5) ──────────────────────────────────

def charger_historique() -> list[dict]:
    if not FICHIER_HISTORIQUE.exists():
        return []
    try:
        with open(FICHIER_HISTORIQUE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        console.print("[yellow]Historique corrompu, réinitialisé.[/yellow]")
        return []

def sauvegarder_historique(historique: list[dict]) -> None:
    FICHIER_HISTORIQUE.parent.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_HISTORIQUE, "w", encoding="utf-8") as f:
        json.dump(historique, f, ensure_ascii=False, indent=2)

def ajouter_message(historique: list[dict], role: str, contenu: str) -> None:
    historique.append({
        "role": role,
        "content": contenu,
        "timestamp": datetime.now().isoformat(),
    })

def messages_pour_api(historique: list[dict]) -> list[dict]:
    """Transforme l'historique (avec timestamp) en format attendu par l'API."""
    return [{"role": m["role"], "content": m["content"]} for m in historique]


# ─── APPEL AU LLM (leçons 3 + 4) ─────────────────────────────────────────────

def creer_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        console.print("[bold red]Erreur : OPENROUTER_API_KEY manquante dans .env[/bold red]")
        sys.exit(1)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

def obtenir_reponse(client: OpenAI, historique: list[dict]) -> str:
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += messages_pour_api(historique)

        response = client.chat.completions.create(
            model=MODELE,
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        return f"[Erreur API] {e}"


# ─── COMMANDES SPÉCIALES (leçon 2) ────────────────────────────────────────────

def afficher_aide() -> None:
    console.print(Panel(
        "[cyan]/aide[/cyan]      → afficher cette aide\n"
        "[cyan]/historique[/cyan] → voir la conversation\n"
        "[cyan]/effacer[/cyan]   → effacer l'historique\n"
        "[cyan]/quitter[/cyan]   → quitter le chatbot",
        title="Commandes disponibles",
    ))

def afficher_historique(historique: list[dict]) -> None:
    if not historique:
        console.print("[dim]Aucun message dans l'historique.[/dim]")
        return
    for msg in historique:
        role = msg["role"]
        contenu = msg["content"]
        heure = msg.get("timestamp", "")[:19].replace("T", " ")
        if role == "user":
            console.print(f"[bold blue]Toi[/bold blue] [dim]({heure})[/dim]: {contenu}")
        else:
            console.print(f"[bold green]Bot[/bold green] [dim]({heure})[/dim]: {contenu}")

def traiter_commande(commande: str, historique: list[dict]) -> bool:
    """Retourne True si c'était une commande (et non un message normal)."""
    if commande == "/aide":
        afficher_aide()
        return True
    if commande == "/historique":
        afficher_historique(historique)
        return True
    if commande == "/effacer":
        historique.clear()
        sauvegarder_historique(historique)
        console.print("[yellow]Historique effacé.[/yellow]")
        return True
    if commande in ("/quitter", "/exit", "/q"):
        sauvegarder_historique(historique)
        console.print("[bold]À bientôt ![/bold]")
        sys.exit(0)
    return False


# ─── BOUCLE PRINCIPALE (leçon 2) ──────────────────────────────────────────────

def lancer_chatbot() -> None:
    console.print(Panel(
        "[bold green]Chatbot IA — Module 1[/bold green]\n"
        f"Modèle : [cyan]{MODELE}[/cyan]\n"
        "Tape [cyan]/aide[/cyan] pour voir les commandes",
        title="Bienvenue",
    ))

    client = creer_client()
    historique = charger_historique()

    if historique:
        console.print(f"[dim]{len(historique)} messages chargés depuis l'historique.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]Toi[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            sauvegarder_historique(historique)
            console.print("\n[bold]À bientôt ![/bold]")
            break

        if not user_input:
            continue

        if traiter_commande(user_input, historique):
            continue

        # Message normal → appel au LLM
        ajouter_message(historique, "user", user_input)

        with console.status("[dim]Réponse en cours...[/dim]"):
            reponse = obtenir_reponse(client, historique)

        ajouter_message(historique, "assistant", reponse)
        sauvegarder_historique(historique)

        console.print(f"[bold green]Bot[/bold green]: {reponse}\n")


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    lancer_chatbot()
