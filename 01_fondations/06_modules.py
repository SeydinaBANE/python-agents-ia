"""
MODULE 1 — Leçon 6 : Modules & Packages
=========================================
Organiser son code en fichiers séparés et utiliser
les bibliothèques installées via uv/pip.
"""

# ─── IMPORTER DES MODULES STANDARD ───────────────────────────────────────────

import os
import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# os — interagir avec le système d'exploitation
print(os.getcwd())                  # répertoire courant
print(os.getenv("HOME"))           # variable d'environnement
print(os.getenv("OPENROUTER_API_KEY", "non définie"))  # valeur par défaut

# datetime — manipuler les dates
maintenant = datetime.now()
print(maintenant.strftime("%Y-%m-%d %H:%M:%S"))

hier = maintenant - timedelta(days=1)
print(hier.date())

# random — pour les tests et simulations
print(random.choice(["claude", "gpt-4", "mistral"]))
print(random.randint(1, 100))


# ─── IMPORTER DEPUIS DES FICHIERS DU PROJET ───────────────────────────────────
# Quand tu as ton propre fichier, tu l'importes par son nom (sans .py)
#
# Structure exemple :
#   pro/
#   ├── config.py           ← ton module
#   └── 01_fondations/
#       └── 06_modules.py   ← ce fichier
#
# Pour importer config.py depuis ici :
#   import sys; sys.path.append("..")
#   from config import get_client, MODELS

# ─── IMPORTER DES PACKAGES INSTALLÉS ─────────────────────────────────────────

# python-dotenv — charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()   # lit le fichier .env à la racine du projet

cle = os.getenv("OPENROUTER_API_KEY")
print(f"Clé API présente : {cle is not None}")

# pydantic — valider des données avec des classes typées
from pydantic import BaseModel, Field, field_validator

class ConfigAgent(BaseModel):
    nom: str
    modele: str = "anthropic/claude-sonnet-4-5"
    max_tokens: int = Field(default=4096, ge=1, le=200_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    actif: bool = True

    @field_validator("nom")
    @classmethod
    def nom_non_vide(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le nom ne peut pas être vide")
        return v.strip()

# Pydantic valide automatiquement les types et les contraintes
config = ConfigAgent(nom="Atlas", max_tokens=8192)
print(config)
print(config.model_dump())      # convertir en dict

# Pydantic lève une erreur si les données sont invalides
try:
    mauvaise_config = ConfigAgent(nom="", temperature=5.0)
except Exception as e:
    print(f"Erreur de validation : {e}")

# rich — affichage amélioré dans le terminal
from rich.console import Console
from rich.table import Table
from rich import print as rprint     # remplace print avec coloration

console = Console()

table = Table(title="Modèles disponibles")
table.add_column("Modèle", style="cyan")
table.add_column("Fournisseur", style="green")
table.add_column("Tokens max", style="yellow")

table.add_row("claude-sonnet-4-5", "Anthropic", "200 000")
table.add_row("gpt-4o",           "OpenAI",    "128 000")
table.add_row("mistral-large",    "Mistral",   "128 000")

console.print(table)

rprint("[bold green]Module 6 terminé ![/bold green]")


# ─── CRÉER SON PROPRE MODULE ──────────────────────────────────────────────────
"""
Pour créer un module réutilisable :

1. Crée un fichier   pro/utils.py
2. Mets-y des fonctions
3. Importe-les avec :  from utils import ma_fonction

Pour un package (dossier de modules) :
1. Crée un dossier   pro/outils/
2. Crée le fichier   pro/outils/__init__.py  (peut être vide)
3. Ajoute des fichiers  pro/outils/texte.py, pro/outils/fichiers.py
4. Importe avec :  from outils.texte import ma_fonction
"""

# ─── __name__ == "__main__" ───────────────────────────────────────────────────
# Ce bloc s'exécute SEULEMENT si tu lances ce fichier directement.
# Il ne s'exécute PAS si quelqu'un importe ce fichier.
# Convention universelle en Python.

if __name__ == "__main__":
    console.print("\n[bold]Ce fichier est lancé directement[/bold]")
    config_test = ConfigAgent(nom="Test", modele="mistral/mistral-large")
    console.print(config_test.model_dump())
