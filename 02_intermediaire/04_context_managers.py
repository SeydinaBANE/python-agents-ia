"""
MODULE 2 — Leçon 4 : Context Managers
=======================================
Le mot-clé 'with' garantit que des ressources sont toujours libérées,
même en cas d'erreur. Essentiel pour les connexions, fichiers, timers.
"""

import time
import json
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from typing import Generator

# ─── with — rappel ────────────────────────────────────────────────────────────
# Le cas le plus courant : fichiers
# with garantit que f.close() est appelé même si une exception survient

with open("02_intermediaire/data/test.txt", "w", encoding="utf-8") as f:
    f.write("test")

Path("02_intermediaire/data").mkdir(parents=True, exist_ok=True)

with open("02_intermediaire/data/test.txt", "w", encoding="utf-8") as f:
    f.write("test")


# ─── CRÉER UN CONTEXT MANAGER AVEC UNE CLASSE ─────────────────────────────────
# Implémenter __enter__ et __exit__

class Timer:
    """Mesure le temps d'exécution d'un bloc de code."""

    def __enter__(self):
        self.debut = time.perf_counter()
        return self                 # la valeur assignée par 'as'

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duree = time.perf_counter() - self.debut
        print(f"Durée : {self.duree:.3f}s")
        return False    # False = ne pas supprimer les exceptions


with Timer() as t:
    time.sleep(0.05)
    resultat = sum(range(100_000))

print(f"Résultat : {resultat}, temps mesuré : {t.duree:.3f}s")


class GestionnaireHistorique:
    """
    Context manager pour lire/écrire un historique JSON en toute sécurité.
    Charge au début, sauvegarde automatiquement à la fin.
    """

    def __init__(self, chemin: str):
        self.chemin = Path(chemin)
        self.historique: list[dict] = []

    def __enter__(self) -> list[dict]:
        if self.chemin.exists():
            try:
                with open(self.chemin, "r", encoding="utf-8") as f:
                    self.historique = json.load(f)
            except json.JSONDecodeError:
                self.historique = []
        return self.historique      # la variable après 'as'

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chemin, "w", encoding="utf-8") as f:
            json.dump(self.historique, f, ensure_ascii=False, indent=2)
        if exc_type:
            print(f"Sauvegardé malgré l'erreur : {exc_val}")
        return False


# Usage propre — plus besoin de penser à sauvegarder
with GestionnaireHistorique("02_intermediaire/data/conv.json") as historique:
    historique.append({"role": "user", "content": "Bonjour"})
    historique.append({"role": "assistant", "content": "Bonjour !"})
# ← fichier sauvegardé automatiquement ici

with GestionnaireHistorique("02_intermediaire/data/conv.json") as historique:
    print(f"Chargé : {len(historique)} messages")


# ─── CRÉER UN CONTEXT MANAGER AVEC @contextmanager ───────────────────────────
# Plus concis que la classe pour les cas simples

@contextmanager
def mesurer(label: str) -> Generator[None, None, None]:
    """Timer simple avec un label."""
    debut = time.perf_counter()
    try:
        yield                   # le code 'with' s'exécute ici
    finally:
        duree = time.perf_counter() - debut
        print(f"[{label}] {duree:.3f}s")


with mesurer("calcul lourd"):
    total = sum(i**2 for i in range(100_000))


@contextmanager
def client_llm_simule(modele: str):
    """Simule l'ouverture/fermeture d'un client LLM."""
    print(f"Connexion à {modele}...")
    client = {"modele": modele, "actif": True}
    try:
        yield client
    finally:
        client["actif"] = False
        print(f"Client {modele} fermé.")


with client_llm_simule("claude-sonnet") as client:
    print(f"Appel avec client actif={client['actif']}")

print(f"Après le bloc : actif={client['actif']}")


# ─── CONTEXT MANAGERS IMBRIQUÉS ───────────────────────────────────────────────

chemin_in  = Path("02_intermediaire/data/test.txt")
chemin_out = Path("02_intermediaire/data/test_out.txt")

with open(chemin_in, "w", encoding="utf-8") as f:
    f.write("bonjour monde")

# Python 3 : plusieurs context managers sur une ligne
with open(chemin_in, "r", encoding="utf-8") as fin, \
     open(chemin_out, "w", encoding="utf-8") as fout:
    contenu = fin.read()
    fout.write(contenu.upper())

print(chemin_out.read_text())   # BONJOUR MONDE


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée un context manager 'barre_chargement(label)' qui affiche
   "⏳ label..." à l'entrée et "✓ label (Xs)" à la sortie.

2. Crée un context manager 'capture_erreurs(fallback)' qui :
   - Attrape toute exception dans le bloc
   - Affiche l'erreur sans la propager
   - Retourne la valeur 'fallback' via une liste [fallback]
     (trick : with capture_erreurs("default") as resultat: ...)

3. Transforme GestionnaireHistorique en @contextmanager (version fonctionnelle).
"""
