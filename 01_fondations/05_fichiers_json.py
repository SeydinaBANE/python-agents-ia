"""
MODULE 1 — Leçon 5 : Fichiers & JSON
======================================
Les agents stockent leur historique, leurs configs, leurs mémoires.
Tout ça passe par des fichiers JSON. C'est la leçon la plus pratique.
"""

import json
from pathlib import Path
from datetime import datetime

# ─── PATHLIB — manipuler les chemins de fichiers ──────────────────────────────
# Préfère pathlib à os.path — c'est plus lisible et portable (Mac/Windows/Linux)

dossier = Path("01_fondations") / "data"
dossier.mkdir(exist_ok=True)        # crée le dossier si absent

fichier = dossier / "test.txt"

# Écrire dans un fichier
fichier.write_text("Bonjour depuis Python !", encoding="utf-8")

# Lire un fichier
contenu = fichier.read_text(encoding="utf-8")
print(contenu)

# Vérifier l'existence
print(fichier.exists())     # True
print(fichier.suffix)       # ".txt"
print(fichier.stem)         # "test"
print(fichier.parent)       # 01_fondations/data


# ─── OPEN() — contrôle fin sur la lecture/écriture ────────────────────────────

chemin = dossier / "notes.txt"

# Écriture ("w" = write, écrase le fichier existant)
with open(chemin, "w", encoding="utf-8") as f:
    f.write("Ligne 1\n")
    f.write("Ligne 2\n")

# Ajout à la fin ("a" = append, ne détruit pas l'existant)
with open(chemin, "a", encoding="utf-8") as f:
    f.write("Ligne 3\n")

# Lecture ("r" = read)
with open(chemin, "r", encoding="utf-8") as f:
    lignes = f.readlines()      # liste de lignes
    print(lignes)

# Toujours utiliser 'with' — il ferme le fichier automatiquement même en cas d'erreur


# ─── JSON — sauvegarder et charger des données structurées ────────────────────

# Un historique de conversation comme le stocke un agent
historique = [
    {"role": "user",      "content": "Bonjour, qui es-tu ?"},
    {"role": "assistant", "content": "Je suis un agent IA."},
    {"role": "user",      "content": "Que sais-tu faire ?"},
]

# Sauvegarder en JSON
chemin_json = dossier / "historique.json"

with open(chemin_json, "w", encoding="utf-8") as f:
    json.dump(historique, f, ensure_ascii=False, indent=2)
    # ensure_ascii=False → garde les caractères accentués lisibles
    # indent=2 → formatage indenté (lisible par un humain)

# Charger depuis JSON
with open(chemin_json, "r", encoding="utf-8") as f:
    historique_charge = json.load(f)

print(historique_charge[0]["content"])  # "Bonjour, qui es-tu ?"

# Convertir dict ↔ string JSON (sans passer par un fichier)
agent = {"nom": "Atlas", "actif": True}
json_str = json.dumps(agent)            # dict → string
print(json_str)                         # '{"nom": "Atlas", "actif": true}'
agent_recharge = json.loads(json_str)   # string → dict
print(agent_recharge["nom"])


# ─── PATTERN : gestionnaire d'historique réutilisable ─────────────────────────

class HistoriqueManager:
    def __init__(self, chemin: str):
        self.chemin = Path(chemin)

    def charger(self) -> list[dict]:
        if not self.chemin.exists():
            return []
        try:
            with open(self.chemin, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def sauvegarder(self, historique: list[dict]) -> None:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chemin, "w", encoding="utf-8") as f:
            json.dump(historique, f, ensure_ascii=False, indent=2)

    def ajouter(self, role: str, contenu: str) -> list[dict]:
        historique = self.charger()
        historique.append({
            "role": role,
            "content": contenu,
            "timestamp": datetime.now().isoformat(),
        })
        self.sauvegarder(historique)
        return historique


# Test du gestionnaire
manager = HistoriqueManager("01_fondations/data/conversation.json")
manager.ajouter("user", "Qu'est-ce qu'un agent IA ?")
manager.ajouter("assistant", "Un agent IA est un programme capable de prendre des décisions.")
manager.ajouter("user", "Donne-moi un exemple.")

historique = manager.charger()
print(f"\n{len(historique)} messages dans l'historique")
for msg in historique:
    print(f"  [{msg['role']}] {msg['content'][:50]}")


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Modifie HistoriqueManager pour ajouter une méthode 'effacer()'
   qui supprime le fichier JSON si il existe.

2. Ajoute une méthode 'dernier_message()' qui retourne le dernier message
   ou None si l'historique est vide.

3. Ajoute une méthode 'filtrer_par_role(role)' qui retourne seulement
   les messages d'un rôle donné ("user" ou "assistant").
"""
