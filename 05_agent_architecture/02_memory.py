"""
MODULE 5 — Leçon 2 : Mémoire des agents
==========================================
Un agent a deux types de mémoire :

  - Mémoire courte (working memory) : l'historique de la conversation en cours.
    Limité par la fenêtre de contexte du LLM (~200k tokens pour Claude).

  - Mémoire longue (long-term memory) : faits persistants entre les sessions.
    Stockés dans un fichier JSON ou une base vectorielle.

Au Module 7, on remplacera la mémoire longue JSON par un vrai vector store.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()


# ─── 1. MÉMOIRE COURTE — historique de la conversation ───────────────────────

class MemoireCourte:
    """
    Gère la fenêtre de contexte envoyée au LLM à chaque appel.
    Inclut une stratégie de troncature pour ne pas dépasser le context window.
    """

    def __init__(self, system_prompt: str, max_messages: int = 20):
        self.system_prompt = system_prompt
        self.max_messages  = max_messages
        self._messages: list[dict] = []

    def ajouter(self, role: str, content: Any) -> None:
        self._messages.append({"role": role, "content": content})

    def pour_api(self) -> list[dict]:
        """Retourne les messages prêts à être envoyés à l'API."""
        msgs = self._messages

        # Troncature : garde les max_messages les plus récents
        # mais conserve TOUJOURS le premier message (contexte initial)
        if len(msgs) > self.max_messages:
            msgs = msgs[:1] + msgs[-(self.max_messages - 1):]

        return [{"role": "system", "content": self.system_prompt}] + msgs

    def effacer(self) -> None:
        self._messages.clear()

    def dernier_message_utilisateur(self) -> str:
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                return msg["content"] if isinstance(msg["content"], str) else ""
        return ""

    @property
    def nb_messages(self) -> int:
        return len(self._messages)

    @property
    def tokens_approx(self) -> int:
        total = len(self.system_prompt)
        for m in self._messages:
            c = m["content"]
            total += len(c) if isinstance(c, str) else len(json.dumps(c))
        return total // 4


# ─── 2. MÉMOIRE LONGUE — faits persistants entre sessions ────────────────────

class Souvenir(BaseModel):
    id:         str      = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    contenu:    str
    categorie:  str      = "general"   # "utilisateur", "préférence", "fait", "contexte"
    importance: int      = Field(default=3, ge=1, le=5)   # 1=faible, 5=critique
    cree_le:    str      = Field(default_factory=lambda: datetime.now().isoformat()[:19])
    utilise:    int      = 0    # nombre de fois injecté dans un prompt


class MemoireLongue:
    """
    Mémoire persistante entre les sessions — stockée en JSON.
    Au Module 7, on la remplace par ChromaDB/pgvector pour la recherche sémantique.
    """

    def __init__(self, chemin: str = "05_agent_architecture/data/memoire.json"):
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self._souvenirs: list[Souvenir] = self._charger()

    def _charger(self) -> list[Souvenir]:
        if not self.chemin.exists():
            return []
        try:
            data = json.loads(self.chemin.read_text(encoding="utf-8"))
            return [Souvenir(**s) for s in data]
        except Exception:
            return []

    def sauvegarder(self) -> None:
        data = [s.model_dump() for s in self._souvenirs]
        self.chemin.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def memoriser(self, contenu: str, categorie: str = "general", importance: int = 3) -> Souvenir:
        souvenir = Souvenir(contenu=contenu, categorie=categorie, importance=importance)
        self._souvenirs.append(souvenir)
        self.sauvegarder()
        return souvenir

    def oublier(self, souvenir_id: str) -> bool:
        avant = len(self._souvenirs)
        self._souvenirs = [s for s in self._souvenirs if s.id != souvenir_id]
        self.sauvegarder()
        return len(self._souvenirs) < avant

    def rechercher(self, query: str, n: int = 5) -> list[Souvenir]:
        """Recherche textuelle simple (remplacée par vector search au Module 7)."""
        query_lower = query.lower()
        pertinents = [
            s for s in self._souvenirs
            if query_lower in s.contenu.lower() or query_lower in s.categorie.lower()
        ]
        # Trie par importance décroissante
        pertinents.sort(key=lambda s: s.importance, reverse=True)
        return pertinents[:n]

    def contexte_pour_prompt(self, query: str = "", n: int = 5) -> str:
        """Génère un bloc de texte à injecter dans le system prompt."""
        if query:
            souvenirs = self.rechercher(query, n)
        else:
            # Sans query : prend les plus importants
            souvenirs = sorted(self._souvenirs, key=lambda s: s.importance, reverse=True)[:n]

        if not souvenirs:
            return ""

        for s in souvenirs:
            s.utilise += 1
        self.sauvegarder()

        lignes = ["## Ce que je sais sur l'utilisateur :"]
        for s in souvenirs:
            lignes.append(f"- [{s.categorie}] {s.contenu}")
        return "\n".join(lignes)

    def afficher(self) -> None:
        table = Table(title=f"Mémoire longue ({len(self._souvenirs)} souvenirs)")
        table.add_column("ID",         style="dim",    width=10)
        table.add_column("Contenu",    style="white",  min_width=40)
        table.add_column("Catégorie",  style="cyan",   width=14)
        table.add_column("Importance", style="yellow", width=11)
        table.add_column("Utilisé",    style="dim",    width=8)

        for s in sorted(self._souvenirs, key=lambda x: x.importance, reverse=True):
            etoiles = "★" * s.importance + "☆" * (5 - s.importance)
            table.add_row(s.id, s.contenu[:60], s.categorie, etoiles, str(s.utilise))

        console.print(table)


# ─── 3. MÉMOIRE COMPLÈTE — combine courte + longue ───────────────────────────

class MemoireAgent:
    """Façade qui combine mémoire courte et longue."""

    def __init__(self, system_prompt: str, chemin_long: str = "05_agent_architecture/data/memoire.json"):
        self.courte = MemoireCourte(system_prompt)
        self.longue = MemoireLongue(chemin_long)

    def construire_system_prompt(self, question: str = "") -> str:
        """System prompt de base enrichi avec les souvenirs pertinents."""
        contexte = self.longue.contexte_pour_prompt(query=question, n=5)
        if not contexte:
            return self.courte.system_prompt
        return self.courte.system_prompt + "\n\n" + contexte

    def messages_pour_api(self, question: str = "") -> list[dict]:
        """Messages complets avec system prompt enrichi."""
        system = self.construire_system_prompt(question)
        msgs   = self.courte.pour_api()
        msgs[0] = {"role": "system", "content": system}   # remplace le system de base
        return msgs

    def ajouter_user(self, contenu: str) -> None:
        self.courte.ajouter("user", contenu)

    def ajouter_assistant(self, contenu: str) -> None:
        self.courte.ajouter("assistant", contenu)

    def memoriser(self, fait: str, **kwargs) -> Souvenir:
        return self.longue.memoriser(fait, **kwargs)


# ─── DÉMONSTRATION ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold]Leçon 2 — Mémoire des agents[/bold]")

    # ── Mémoire courte ────────────────────────────────────────────────────────
    console.print("\n[bold cyan]1. Mémoire courte[/bold cyan]")
    mem = MemoireCourte("Tu es un assistant IA.", max_messages=6)

    for i in range(5):
        mem.ajouter("user",      f"Question {i+1}")
        mem.ajouter("assistant", f"Réponse {i+1}")

    console.print(f"Messages total : {mem.nb_messages}")
    console.print(f"Tokens approx  : {mem.tokens_approx}")
    console.print(f"Pour API       : {len(mem.pour_api())} messages (system + {mem.nb_messages} conv)")

    # ── Mémoire longue ────────────────────────────────────────────────────────
    console.print("\n[bold cyan]2. Mémoire longue[/bold cyan]")
    mem_longue = MemoireLongue()

    mem_longue.memoriser("L'utilisateur s'appelle Seydina",           categorie="utilisateur",  importance=5)
    mem_longue.memoriser("Il apprend Python pour créer des agents IA", categorie="contexte",     importance=5)
    mem_longue.memoriser("Il préfère les réponses courtes et directes", categorie="préférence",  importance=4)
    mem_longue.memoriser("Il utilise OpenRouter comme fournisseur LLM", categorie="technique",   importance=3)
    mem_longue.memoriser("Il travaille sur macOS",                     categorie="environnement", importance=2)

    mem_longue.afficher()

    console.print("\n[bold]Recherche 'agent' :[/bold]")
    for s in mem_longue.rechercher("agent", n=3):
        console.print(f"  [{s.categorie}] {s.contenu}")

    console.print("\n[bold]Contexte pour prompt :[/bold]")
    console.print(mem_longue.contexte_pour_prompt("agent IA"))

    # ── Mémoire combinée ──────────────────────────────────────────────────────
    console.print("\n[bold cyan]3. Mémoire combinée[/bold cyan]")
    mem_agent = MemoireAgent("Tu es un assistant IA expert.")
    mem_agent.memoriser("L'utilisateur préfère Python", categorie="préférence", importance=4)
    mem_agent.ajouter_user("Bonjour !")
    mem_agent.ajouter_assistant("Bonjour ! Comment puis-je vous aider ?")

    msgs = mem_agent.messages_pour_api("Python")
    console.print(f"System prompt enrichi :\n[dim]{msgs[0]['content'][:300]}[/dim]")
