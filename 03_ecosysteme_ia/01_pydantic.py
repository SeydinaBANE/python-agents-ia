"""
MODULE 3 — Leçon 1 : Pydantic v2
==================================
Pydantic valide les données à l'entrée et génère les schémas JSON
dont les LLMs ont besoin pour appeler tes tools.
C'est la bibliothèque la plus utilisée dans les agents IA.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError
from typing import Any
from enum import Enum

# ─── BASEMODEL — valider un dict automatiquement ─────────────────────────────

class Message(BaseModel):
    role: str
    content: str

msg = Message(role="user", content="Bonjour")
print(msg)
print(msg.model_dump())             # → dict Python
print(msg.model_dump_json())        # → string JSON

# Pydantic lève ValidationError si le type est mauvais
try:
    Message(role=123, content=None)  # type: ignore
except ValidationError as e:
    print(e)


# ─── FIELD — contraintes précises ────────────────────────────────────────────

class ConfigLLM(BaseModel):
    modele:      str   = Field(default="anthropic/claude-haiku-4-5")
    max_tokens:  int   = Field(default=1024,  ge=1,    le=200_000)
    temperature: float = Field(default=0.7,   ge=0.0,  le=2.0)
    top_p:       float = Field(default=1.0,   ge=0.0,  le=1.0)
    system:      str   = Field(default="Tu es un assistant IA.", min_length=1)
    tags:        list[str] = Field(default_factory=list)

config = ConfigLLM(temperature=0.3, max_tokens=2048)
print(config)

try:
    ConfigLLM(temperature=5.0)  # dépasse le max → erreur
except ValidationError as e:
    print(f"Erreur : {e.errors()[0]['msg']}")


# ─── VALIDATORS — logique de validation custom ────────────────────────────────

class NomAgent(str, Enum):
    ATLAS  = "atlas"
    NOVA   = "nova"
    ORION  = "orion"

class Agent(BaseModel):
    nom:     str
    modele:  str
    budget:  int = Field(default=10_000, description="Budget en tokens")
    actif:   bool = True

    @field_validator("nom")
    @classmethod
    def nom_valide(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le nom ne peut pas être vide")
        return v.title()                # "seydina" → "Seydina"

    @field_validator("modele")
    @classmethod
    def modele_connu(cls, v: str) -> str:
        prefixes_valides = ("anthropic/", "openai/", "mistralai/", "meta-llama/")
        if not any(v.startswith(p) for p in prefixes_valides):
            raise ValueError(f"Modèle '{v}' non reconnu. Préfixe attendu : {prefixes_valides}")
        return v

    @model_validator(mode="after")
    def coherence_budget(self) -> "Agent":
        # Un agent inactif n'a pas besoin de budget
        if not self.actif and self.budget > 0:
            self.budget = 0
        return self


a = Agent(nom=" seydina ", modele="anthropic/claude-haiku-4-5", budget=5000)
print(a)    # nom="Seydina"

try:
    Agent(nom="", modele="gpt-4")
except ValidationError as e:
    for err in e.errors():
        print(f"  [{err['loc'][0]}] {err['msg']}")


# ─── SCHÉMA JSON — générer le schéma pour l'API tool use ─────────────────────
# OpenAI/OpenRouter demande un JSON Schema pour décrire chaque tool.
# Pydantic le génère automatiquement !

class RechercheInput(BaseModel):
    """Paramètres pour la recherche web."""
    query:        str = Field(description="Terme de recherche")
    max_resultats: int = Field(default=5, ge=1, le=20, description="Nombre max de résultats")
    langue:        str = Field(default="fr", description="Code langue ISO (fr, en, es...)")

schema = RechercheInput.model_json_schema()
print(schema)
# → exactement ce qu'il faut mettre dans "parameters" d'un tool OpenAI


# ─── IMBRICATION DE MODÈLES ───────────────────────────────────────────────────

class Outil(BaseModel):
    nom:         str
    description: str
    parametres:  dict[str, Any] = Field(default_factory=dict)

class AgentComplet(BaseModel):
    nom:        str
    config:     ConfigLLM
    outils:     list[Outil] = Field(default_factory=list)
    historique: list[Message] = Field(default_factory=list)

    def ajouter_message(self, role: str, content: str) -> None:
        self.historique.append(Message(role=role, content=content))

    def dernier_message(self) -> Message | None:
        return self.historique[-1] if self.historique else None


agent = AgentComplet(
    nom="Atlas",
    config=ConfigLLM(temperature=0.5),
    outils=[
        Outil(nom="recherche_web", description="Cherche sur le web"),
        Outil(nom="calculatrice",  description="Effectue des calculs"),
    ],
)
agent.ajouter_message("user", "Bonjour !")
print(agent.dernier_message())
print(agent.model_dump_json(indent=2))


# ─── SÉRIALISATION / DÉSÉRIALISATION ─────────────────────────────────────────

import pathlib

# Sauvegarder
data = agent.model_dump_json(indent=2)
pathlib.Path("03_ecosysteme_ia/data").mkdir(exist_ok=True)
pathlib.Path("03_ecosysteme_ia/data/agent.json").write_text(data)

# Recharger
raw = pathlib.Path("03_ecosysteme_ia/data/agent.json").read_text()
agent_recharge = AgentComplet.model_validate_json(raw)
print(f"Rechargé : {agent_recharge.nom}, {len(agent_recharge.outils)} outils")


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée un modèle 'AppelTool' avec :
   - id (str, généré automatiquement avec uuid4 comme default_factory)
   - nom (str)
   - arguments (dict[str, Any])
   - timestamp (datetime, default = maintenant)

2. Crée un modèle 'ReponseAPI' qui parse la réponse brute d'OpenRouter :
   {
     "id": "...",
     "model": "...",
     "choices": [{"message": {"role": "...", "content": "..."}}],
     "usage": {"prompt_tokens": 10, "completion_tokens": 20}
   }

3. Génère et affiche le JSON Schema de 'AppelTool' — c'est exactement
   ce qu'un agent envoie au LLM pour lui décrire un tool.
"""
