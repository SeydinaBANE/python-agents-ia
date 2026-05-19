"""
MODULE 1 — Leçon 1 : Types & Variables
======================================
En Python, pas besoin de déclarer le type d'une variable.
Python le devine tout seul. C'est ce qu'on appelle le "typage dynamique".
"""

# ─── LES TYPES DE BASE ────────────────────────────────────────────────────────

# int — nombre entier
age = 25
print(type(age))        # <class 'int'>

# float — nombre décimal
prix = 9.99
print(type(prix))       # <class 'float'>

# str — chaîne de caractères (toujours entre guillemets)
nom = "Seydina"
prenom = 'Bane'         # guillemets simples ou doubles, au choix
message = f"Bonjour {nom} {prenom} !"   # f-string : intègre des variables
print(message)

# bool — vrai ou faux (commence par une majuscule en Python !)
est_connecte = True
est_admin = False
print(type(est_connecte))   # <class 'bool'>

# None — l'absence de valeur (équivalent de null dans d'autres langages)
resultat = None
print(resultat)             # None


# ─── LES COLLECTIONS ──────────────────────────────────────────────────────────

# list — liste ordonnée, modifiable, peut contenir des types mixtes
fruits = ["pomme", "banane", "cerise"]
fruits.append("mangue")        # ajouter à la fin
fruits.remove("banane")        # supprimer un élément
print(fruits[0])               # accès par index (commence à 0)
print(fruits[-1])              # dernier élément
print(fruits[1:3])             # slice : éléments 1 et 2 (3 exclu)

# tuple — liste ordonnée, NON modifiable
coordonnees = (48.8566, 2.3522)  # latitude/longitude de Paris
# coordonnees[0] = 0  ← erreur ! un tuple ne change pas
print(coordonnees[0])

# dict — dictionnaire : paires clé → valeur
agent = {
    "nom": "GPT",
    "version": 4,
    "actif": True,
    "outils": ["recherche", "calcul"],
}
print(agent["nom"])             # accéder à une clé
agent["temperature"] = 0.7     # ajouter une clé
print(agent.get("modele", "inconnu"))  # .get() évite les erreurs si clé absente

# set — ensemble, sans doublons, non ordonné
tags = {"python", "ia", "agent", "python"}  # "python" en double → ignoré
print(tags)                     # {"python", "ia", "agent"}
tags.add("llm")


# ─── OPÉRATIONS UTILES ────────────────────────────────────────────────────────

# Vérifier si une valeur existe
print("pomme" in fruits)        # True
print("raisin" in fruits)       # False

# Longueur d'une collection
print(len(fruits))              # 3
print(len(agent))               # 5

# Convertir des types
nombre_str = "42"
nombre_int = int(nombre_str)    # str → int
nombre_float = float("3.14")   # str → float
retour_str = str(nombre_int)    # int → str
print(type(nombre_int))         # <class 'int'>


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée un dictionnaire 'mon_agent' avec les clés :
   - "nom" (str), "modele" (str), "max_tokens" (int), "actif" (bool)

2. Ajoute une clé "historique" avec une liste vide []

3. Affiche le type de chaque valeur avec type()

4. Vérifie si la clé "temperature" est dans le dictionnaire
"""


# ─── SOLUTION ─────────────────────────────────────────────────────────────────

mon_agent = {
    "nom": "MonAgent",
    "modele": "claude-sonnet",
    "max_tokens": 4096,
    "actif": True,
}

mon_agent["historique"] = []

for cle, valeur in mon_agent.items():
    print(f"{cle}: {valeur} → {type(valeur).__name__}")

print("temperature" in mon_agent)   # False
