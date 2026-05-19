"""
MODULE 5 — Leçon 4 : Planification et décomposition de tâches
===============================================================
Pour les tâches complexes, un agent doit d'abord créer un plan,
puis exécuter chaque étape. C'est ce qu'on appelle le pattern
"Plan and Execute" ou "Task Decomposition".

Flux :
  1. Planifier  : le LLM décompose la question en sous-tâches
  2. Exécuter   : chaque sous-tâche est traitée (outil ou LLM)
  3. Synthétiser : le LLM combine les résultats en réponse finale
"""

import asyncio
import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()
console = Console()
MODELE  = "anthropic/claude-sonnet-4-5"


# ─── MODÈLES DE DONNÉES ───────────────────────────────────────────────────────

class SousTache(BaseModel):
    id:          int
    description: str
    type:        Literal["outil", "llm"]   # outil = appel tool, llm = réponse directe
    outil:       str | None  = None
    arguments:   dict        = Field(default_factory=dict)
    dependances: list[int]   = Field(default_factory=list)   # IDs des tâches à terminer avant
    resultat:    str | None  = None
    terminee:    bool        = False

class Plan(BaseModel):
    question:    str
    taches:      list[SousTache]
    synthese:    str | None  = None

    @property
    def toutes_terminees(self) -> bool:
        return all(t.terminee for t in self.taches)

    @property
    def prochaines(self) -> list[SousTache]:
        """Tâches dont toutes les dépendances sont satisfaites."""
        terminees = {t.id for t in self.taches if t.terminee}
        return [
            t for t in self.taches
            if not t.terminee and all(d in terminees for d in t.dependances)
        ]


# ─── OUTILS DISPONIBLES ──────────────────────────────────────────────────────

def get_meteo(ville: str) -> dict:
    meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Berlin": "12°C froid"}
    return {"ville": ville, "meteo": meteos.get(ville, "20°C inconnu")}

def calculer(expression: str) -> dict:
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Non autorisé"}
        return {"resultat": eval(expression)}  # noqa: S307
    except Exception as e:
        return {"erreur": str(e)}

def chercher(sujet: str) -> dict:
    return {"resultats": [f"Résultat sur '{sujet}' #{i+1}" for i in range(3)]}

OUTILS_FN = {"get_meteo": get_meteo, "calculer": calculer, "chercher": chercher}


# ─── AGENT PLANIFICATEUR ─────────────────────────────────────────────────────

class AgentPlanificateur:

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # ── Étape 1 : Créer le plan ───────────────────────────────────────────────

    async def planifier(self, question: str) -> Plan:
        """Demande au LLM de décomposer la question en sous-tâches."""
        schema = Plan.model_json_schema()

        response = await self.client.chat.completions.create(
            model=MODELE,
            messages=[
                {
                    "role": "system",
                    "content": f"""Tu es un planificateur d'agent IA.
Décompose la question en sous-tâches ordonnées. Outils disponibles : get_meteo, calculer, chercher.
Réponds UNIQUEMENT en JSON selon ce schéma (sans markdown) :
{json.dumps(schema, indent=2, ensure_ascii=False)}

Règle : si une tâche B dépend du résultat de A, mets l'ID de A dans dependances de B.""",
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
        )

        raw  = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        data["question"] = question
        return Plan.model_validate(data)

    # ── Étape 2 : Exécuter une sous-tâche ─────────────────────────────────────

    async def executer_tache(self, tache: SousTache, contexte: str = "") -> str:
        """Exécute une sous-tâche — soit via outil, soit via LLM."""
        if tache.type == "outil" and tache.outil:
            fn = OUTILS_FN.get(tache.outil)
            if fn:
                resultat = fn(**tache.arguments)
                return json.dumps(resultat, ensure_ascii=False)
            return json.dumps({"erreur": f"Outil '{tache.outil}' inconnu"})

        # Type "llm" : on demande directement au LLM
        response = await self.client.chat.completions.create(
            model=MODELE,
            messages=[
                {"role": "system", "content": "Réponds de façon concise."},
                {"role": "user",   "content": f"{contexte}\n\nTâche : {tache.description}"},
            ],
            max_tokens=256,
        )
        return response.choices[0].message.content or ""

    # ── Étape 3 : Synthétiser ─────────────────────────────────────────────────

    async def synthetiser(self, plan: Plan) -> str:
        """Combine tous les résultats en une réponse finale cohérente."""
        resultats_text = "\n".join(
            f"- {t.description} → {t.resultat}"
            for t in plan.taches if t.resultat
        )

        response = await self.client.chat.completions.create(
            model=MODELE,
            messages=[
                {"role": "system", "content": "Tu synthétises des résultats en une réponse claire et concise."},
                {"role": "user",   "content": f"Question originale : {plan.question}\n\nRésultats obtenus :\n{resultats_text}\n\nSynthétise en une réponse finale."},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    # ── Orchestration complète ────────────────────────────────────────────────

    async def resoudre(self, question: str, verbose: bool = True) -> str:
        """Pipeline complet : plan → exécution → synthèse."""

        # 1. Planification
        if verbose:
            console.print(f"\n[bold blue]Question :[/bold blue] {question}")

        with console.status("[dim]Planification...[/dim]"):
            plan = await self.planifier(question)

        if verbose:
            console.print(f"\n[bold]Plan : {len(plan.taches)} tâche(s)[/bold]")
            for t in plan.taches:
                deps = f" (après {t.dependances})" if t.dependances else ""
                console.print(f"  [cyan]{t.id}.[/cyan] [{t.type}] {t.description}{deps}")

        # 2. Exécution — en respectant les dépendances
        iterations = 0
        while not plan.toutes_terminees and iterations < 10:
            iterations += 1
            prochaines = plan.prochaines

            if not prochaines:
                break

            # Exécuter les tâches prêtes en parallèle
            contexte = "\n".join(
                f"Résultat tâche {t.id} : {t.resultat}"
                for t in plan.taches if t.terminee and t.resultat
            )

            resultats = await asyncio.gather(*[
                self.executer_tache(t, contexte) for t in prochaines
            ])

            for tache, resultat in zip(prochaines, resultats):
                tache.resultat  = resultat
                tache.terminee  = True
                if verbose:
                    console.print(f"  [green]✓[/green] Tâche {tache.id} : [dim]{resultat[:80]}[/dim]")

        # 3. Synthèse
        with console.status("[dim]Synthèse...[/dim]"):
            synthese = await self.synthetiser(plan)

        plan.synthese = synthese
        if verbose:
            console.print(Panel(synthese, title="[bold green]Réponse finale[/bold green]"))

        return synthese


# ─── DÉMONSTRATION ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 4 — Planification[/bold]")
    agent = AgentPlanificateur()

    await agent.resoudre(
        "Quel temps fait-il à Paris, Dakar et Berlin ? "
        "Quelle ville est la plus chaude ? Calcule la différence de température entre la plus chaude et la plus froide."
    )


if __name__ == "__main__":
    asyncio.run(main())
