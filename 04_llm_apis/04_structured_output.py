"""
MODULE 4 — Leçon 4 : Structured Output
=========================================
Forcer le LLM à répondre dans un format JSON précis.
Essentiel pour les agents : extraire des données, classifier,
produire des résultats que du code Python peut consommer.

Deux approches :
  1. response_format={"type": "json_object"} — JSON libre
  2. Pydantic + prompt — JSON structuré et validé
"""

import asyncio
import json
import os
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
MODELE = "anthropic/claude-sonnet-4-5"

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


# ─── 1. JSON MODE ─────────────────────────────────────────────────────────────
# response_format={"type":"json_object"} garantit que la réponse est du JSON valide.
# Le prompt doit demander explicitement du JSON et décrire le format.

async def extraire_json_libre(texte: str) -> dict:
    client = get_client()
    response = await client.chat.completions.create(
        model=MODELE,
        messages=[
            {
                "role": "system",
                "content": "Tu es un extracteur de données. Réponds UNIQUEMENT en JSON valide, sans markdown.",
            },
            {
                "role": "user",
                "content": f"""Extrais les informations de ce texte et retourne un JSON avec les clés :
nom, age, ville, profession.

Texte : {texte}""",
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


# ─── 2. STRUCTURED OUTPUT AVEC PYDANTIC ──────────────────────────────────────
# Approche plus robuste : on définit le schéma avec Pydantic,
# on le convertit en JSON Schema, et on l'inclut dans le prompt.

class AnalyseSentiment(BaseModel):
    sentiment:  str   = Field(description="positif, négatif, ou neutre")
    score:      float = Field(description="Score de 0.0 (très négatif) à 1.0 (très positif)", ge=0.0, le=1.0)
    mots_cles:  list[str] = Field(description="3 mots clés du texte")
    resume:     str   = Field(description="Résumé en une phrase")


class ActionAgent(BaseModel):
    """Représente la décision structurée d'un agent."""
    pensee:    str   = Field(description="Raisonnement interne de l'agent")
    action:    str   = Field(description="Action à effectuer : 'rechercher', 'calculer', 'repondre', 'demander_clarification'")
    parametre: str   = Field(description="Paramètre de l'action")
    confiance: float = Field(description="Niveau de confiance de 0 à 1", ge=0.0, le=1.0)


async def analyser_avec_schema(texte: str, schema_model: type[BaseModel]) -> BaseModel:
    """Demande au LLM de répondre selon un schéma Pydantic précis."""
    client = get_client()
    schema = schema_model.model_json_schema()

    response = await client.chat.completions.create(
        model=MODELE,
        messages=[
            {
                "role": "system",
                "content": f"""Tu dois répondre UNIQUEMENT avec un objet JSON valide qui respecte ce schéma :

{json.dumps(schema, indent=2, ensure_ascii=False)}

Pas de texte avant ou après le JSON. Pas de markdown. Juste le JSON.""",
            },
            {"role": "user", "content": texte},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
    )

    raw = response.choices[0].message.content or "{}"
    return schema_model.model_validate_json(raw)


# ─── 3. EXTRACTION DE DONNÉES STRUCTURÉES ────────────────────────────────────

class Produit(BaseModel):
    nom:       str
    prix:      float
    devise:    str = "EUR"
    en_stock:  bool
    categorie: str

class ListeProduits(BaseModel):
    produits: list[Produit]
    total:    int = Field(description="Nombre total de produits trouvés")

async def extraire_produits(texte: str) -> ListeProduits:
    client = get_client()
    schema = ListeProduits.model_json_schema()

    response = await client.chat.completions.create(
        model=MODELE,
        messages=[
            {
                "role": "system",
                "content": f"Extrais tous les produits mentionnés. Réponds en JSON selon ce schéma :\n{json.dumps(schema, indent=2)}",
            },
            {"role": "user", "content": texte},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
    )
    raw = response.choices[0].message.content or "{}"
    return ListeProduits.model_validate_json(raw)


# ─── 4. CHAIN OF THOUGHT STRUCTURÉ ───────────────────────────────────────────
# Forcer l'agent à raisonner étape par étape avant de répondre

class RaisonnementEtape(BaseModel):
    etape:       int
    description: str
    resultat:    str

class RaisonnementComplet(BaseModel):
    question:  str
    etapes:    list[RaisonnementEtape]
    conclusion: str
    confiance:  float = Field(ge=0.0, le=1.0)

async def raisonner_etape_par_etape(question: str) -> RaisonnementComplet:
    client = get_client()
    schema = RaisonnementComplet.model_json_schema()

    response = await client.chat.completions.create(
        model=MODELE,
        messages=[
            {
                "role": "system",
                "content": f"""Raisonne étape par étape et retourne ta réflexion en JSON :
{json.dumps(schema, indent=2)}""",
            },
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    raw = response.choices[0].message.content or "{}"
    return RaisonnementComplet.model_validate_json(raw)


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

async def main():
    console.rule("[bold]Leçon 4 — Structured Output[/bold]")

    # Test 1 : extraction JSON libre
    console.print("\n[bold cyan]1. Extraction JSON libre[/bold cyan]")
    info = await extraire_json_libre(
        "Marie Dupont, 34 ans, vit à Bordeaux et travaille comme développeuse."
    )
    console.print(info)

    # Test 2 : analyse de sentiment
    console.print("\n[bold cyan]2. Analyse de sentiment (Pydantic)[/bold cyan]")
    analyse = await analyser_avec_schema(
        "Ce framework est absolument incroyable ! J'ai pu créer mon premier agent en 30 minutes.",
        AnalyseSentiment,
    )
    console.print(analyse.model_dump())

    # Test 3 : extraction de produits
    console.print("\n[bold cyan]3. Extraction de produits[/bold cyan]")
    texte_produits = """
    Notre boutique propose : un ordinateur portable Dell à 899€ (en stock),
    un clavier mécanique Keychron à 129€ (rupture de stock),
    et une souris Logitech MX Master à 89€ (en stock).
    """
    produits = await extraire_produits(texte_produits)
    for p in produits.produits:
        stock = "[green]✓[/green]" if p.en_stock else "[red]✗[/red]"
        console.print(f"  {stock} {p.nom} — {p.prix}{p.devise}")

    # Test 4 : raisonnement structuré
    console.print("\n[bold cyan]4. Chain of Thought structuré[/bold cyan]")
    raisonnement = await raisonner_etape_par_etape(
        "Un agent IA doit répondre à 100 questions. Il traite 10 questions par minute "
        "mais s'arrête 5 minutes toutes les 30 questions. Combien de temps en tout ?"
    )
    for etape in raisonnement.etapes:
        console.print(f"  [yellow]Étape {etape.etape}[/yellow] : {etape.description}")
        console.print(f"    → {etape.resultat}")
    console.print(f"\n[bold green]Conclusion :[/bold green] {raisonnement.conclusion}")
    console.print(f"[dim]Confiance : {raisonnement.confiance:.0%}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
