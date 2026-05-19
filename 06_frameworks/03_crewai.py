"""
MODULE 6 — Leçon 3 : CrewAI
==============================
CrewAI organise plusieurs agents spécialisés en une équipe (crew).
Chaque agent a un rôle, des outils, et un objectif.
Ils collaborent pour accomplir des tâches complexes.

Analogie : une équipe projet avec un chercheur, un analyste, et un rédacteur.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import Field


# ─── 1. CONFIGURER LE LLM OPENROUTER POUR CREWAI ─────────────────────────────

def get_llm(modele: str = "anthropic/claude-haiku-4-5") -> LLM:
    return LLM(
        model=f"openrouter/{modele}",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


# ─── 2. OUTILS PERSONNALISÉS ──────────────────────────────────────────────────
# CrewAI utilise des classes héritant de BaseTool

class OutilMeteo(BaseTool):
    name:        str = "get_meteo"
    description: str = "Retourne la météo d'une ville."

    def _run(self, ville: str) -> str:
        meteos = {"Paris": "18°C nuageux", "Dakar": "32°C ensoleillé", "Lyon": "15°C pluvieux"}
        return meteos.get(ville, f"20°C inconnu pour {ville}")


class OutilCalcul(BaseTool):
    name:        str = "calculer"
    description: str = "Évalue une expression mathématique. Input: expression (ex: '15 * 8')"

    def _run(self, expression: str) -> str:
        try:
            autorise = set("0123456789+-*/()., ")
            if not all(c in autorise for c in expression):
                return "Expression non autorisée"
            return str(eval(expression))  # noqa: S307
        except Exception as e:
            return f"Erreur: {e}"


class OutilRecherche(BaseTool):
    name:        str = "rechercher"
    description: str = "Cherche des informations sur un sujet."

    def _run(self, sujet: str) -> str:
        return f"Résultats sur '{sujet}': [fait 1, fait 2, fait 3 — simulation]"


# ─── 3. CREW SIMPLE : UN SEUL AGENT ──────────────────────────────────────────

def demo_agent_simple(question: str) -> str:
    """Un seul agent avec des outils."""
    llm   = get_llm()
    outils = [OutilMeteo(), OutilCalcul(), OutilRecherche()]

    agent = Agent(
        role="Assistant IA polyvalent",
        goal="Répondre aux questions de l'utilisateur avec précision.",
        backstory="Tu es un assistant expert qui utilise des outils pour répondre.",
        tools=outils,
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    tache = Task(
        description=question,
        expected_output="Une réponse claire et complète en français.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[tache],
        process=Process.sequential,
        verbose=False,
    )

    resultat = crew.kickoff()
    return str(resultat)


# ─── 4. CREW MULTI-AGENTS : ÉQUIPE SPÉCIALISÉE ───────────────────────────────
# Pattern classique : Chercheur → Analyste → Rédacteur

def demo_crew_editorial(sujet: str) -> str:
    """
    Équipe de 3 agents qui travaillent en séquence :
    1. Chercheur : collecte les informations
    2. Analyste  : analyse et structure
    3. Rédacteur : rédige le contenu final
    """
    llm_rapide  = get_llm("anthropic/claude-haiku-4-5")
    llm_qualite = get_llm("anthropic/claude-sonnet-4-5")

    # Agent 1 : Chercheur
    chercheur = Agent(
        role="Chercheur en IA",
        goal="Collecter des informations précises et pertinentes.",
        backstory="Tu es un chercheur expert qui sait trouver les faits importants.",
        tools=[OutilRecherche()],
        llm=llm_rapide,
        verbose=False,
        max_iter=3,
    )

    # Agent 2 : Analyste
    analyste = Agent(
        role="Analyste technique",
        goal="Analyser les informations et en extraire les points clés.",
        backstory="Tu es un analyste qui structure l'information de façon logique.",
        llm=llm_rapide,
        verbose=False,
        max_iter=3,
    )

    # Agent 3 : Rédacteur
    redacteur = Agent(
        role="Rédacteur pédagogique",
        goal="Rédiger du contenu clair et accessible pour des apprenants.",
        backstory="Tu écris pour des développeurs qui découvrent les agents IA.",
        llm=llm_qualite,
        verbose=False,
        max_iter=3,
    )

    # Tâches en séquence
    t1 = Task(
        description=f"Recherche des informations sur : {sujet}. Liste 5 faits importants.",
        expected_output="Une liste de 5 faits clés avec des explications courtes.",
        agent=chercheur,
    )

    t2 = Task(
        description="Analyse les informations collectées. Identifie les 3 points les plus importants et explique pourquoi.",
        expected_output="Un rapport d'analyse avec 3 points clés justifiés.",
        agent=analyste,
        context=[t1],   # dépend de t1
    )

    t3 = Task(
        description="Rédige un mini-guide pédagogique basé sur l'analyse. Formate en Markdown avec des exemples.",
        expected_output="Un mini-guide Markdown de 200 mots maximum avec exemples.",
        agent=redacteur,
        context=[t1, t2],   # dépend de t1 et t2
    )

    crew = Crew(
        agents=[chercheur, analyste, redacteur],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False,
    )

    resultat = crew.kickoff()
    return str(resultat)


# ─── 5. CREW PARALLÈLE ────────────────────────────────────────────────────────
# Plusieurs agents travaillent en même temps sur des aspects différents

def demo_crew_parallele(question: str) -> str:
    """
    Deux agents analysent en parallèle, un troisième synthétise.
    """
    llm = get_llm()

    expert_tech = Agent(
        role="Expert technique",
        goal="Analyser les aspects techniques de la question.",
        backstory="Ingénieur senior spécialisé en systèmes IA.",
        llm=llm, verbose=False, max_iter=3,
    )

    expert_pratique = Agent(
        role="Expert pratique",
        goal="Analyser les cas d'usage pratiques et concrets.",
        backstory="Consultant qui aide les entreprises à déployer des agents IA.",
        llm=llm, verbose=False, max_iter=3,
    )

    synthetiseur = Agent(
        role="Synthétiseur",
        goal="Combiner les perspectives pour donner une réponse complète.",
        backstory="Expert en vulgarisation scientifique.",
        llm=llm, verbose=False, max_iter=3,
    )

    t_tech = Task(
        description=f"Analyse technique de : {question}",
        expected_output="Analyse technique en 3 points.",
        agent=expert_tech,
    )

    t_pratique = Task(
        description=f"Analyse pratique de : {question}",
        expected_output="3 cas d'usage concrets.",
        agent=expert_pratique,
    )

    t_synthese = Task(
        description="Synthétise les deux analyses en une réponse claire et complète.",
        expected_output="Réponse synthétique de 150 mots maximum.",
        agent=synthetiseur,
        context=[t_tech, t_pratique],
    )

    crew = Crew(
        agents=[expert_tech, expert_pratique, synthetiseur],
        tasks=[t_tech, t_pratique, t_synthese],
        process=Process.sequential,
        verbose=False,
    )

    resultat = crew.kickoff()
    return str(resultat)


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold]Leçon 3 — CrewAI[/bold]")

    # Test 1 : agent simple
    console.print("\n[bold cyan]1. Agent simple avec outils[/bold cyan]")
    rep = demo_agent_simple("Quel temps fait-il à Dakar ? Et combien font 42 * 7 ?")
    console.print(Panel(rep, title="Réponse agent simple"))

    # Test 2 : crew éditorial
    console.print("\n[bold cyan]2. Crew éditorial (3 agents séquentiels)[/bold cyan]")
    rep = demo_crew_editorial("les agents IA ReAct")
    console.print(Panel(rep[:500], title="Mini-guide généré"))

    # Test 3 : crew parallèle
    console.print("\n[bold cyan]3. Crew parallèle (analyse + synthèse)[/bold cyan]")
    rep = demo_crew_parallele("Comment les agents IA vont transformer le développement logiciel ?")
    console.print(Panel(rep[:500], title="Synthèse parallèle"))
