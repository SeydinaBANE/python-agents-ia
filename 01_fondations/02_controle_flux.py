"""
MODULE 1 — Leçon 2 : Contrôle de flux
======================================
Comment faire prendre des décisions à ton programme
et répéter des actions.
"""

# ─── IF / ELIF / ELSE ─────────────────────────────────────────────────────────

score = 85

if score >= 90:
    print("Excellent !")
elif score >= 70:
    print("Bien !")        # ← cette ligne s'exécute
elif score >= 50:
    print("Passable")
else:
    print("À revoir")

# Condition sur une ligne (ternaire) — utile pour les cas simples
statut = "actif" if score > 50 else "inactif"
print(statut)

# Combiner des conditions
age = 20
a_carte = True

if age >= 18 and a_carte:
    print("Accès autorisé")

if age < 18 or not a_carte:
    print("Accès refusé")


# ─── FOR — parcourir une collection ───────────────────────────────────────────

modeles = ["gpt-4", "claude-3", "mistral", "llama"]

for modele in modeles:
    print(f"Modèle : {modele}")

# range() — générer une séquence de nombres
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):       # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8  (pas de 2)
    print(i)

# enumerate() — index + valeur en même temps
for index, modele in enumerate(modeles):
    print(f"{index}: {modele}")

# Parcourir un dictionnaire
agent = {"nom": "Atlas", "modele": "claude-sonnet", "tokens": 4096}

for cle, valeur in agent.items():
    print(f"  {cle} = {valeur}")

# List comprehension — créer une liste en une ligne
longueurs = [len(m) for m in modeles]
print(longueurs)    # [5, 8, 7, 5]

grands_modeles = [m for m in modeles if len(m) > 6]
print(grands_modeles)   # ['claude-3', 'mistral']


# ─── WHILE — répéter tant qu'une condition est vraie ──────────────────────────

compteur = 0
while compteur < 3:
    print(f"Tour {compteur + 1}")
    compteur += 1   # IMPORTANT : toujours modifier la condition pour éviter la boucle infinie

# break — sortir d'une boucle
messages = ["bonjour", "comment vas-tu", "quitte", "autre chose"]
for msg in messages:
    if msg == "quitte":
        print("Fin de la conversation")
        break           # on sort immédiatement de la boucle
    print(f"Message reçu : {msg}")

# continue — sauter une itération
for i in range(10):
    if i % 2 == 0:
        continue        # sauter les nombres pairs
    print(i)            # affiche 1, 3, 5, 7, 9


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Écris une boucle qui parcourt une liste de messages et affiche
   "[LONG]" si le message dépasse 50 caractères, sinon "[COURT]"

2. Avec une list comprehension, crée une liste des messages longs uniquement

3. Simule une boucle de chatbot : demande un input à l'utilisateur
   en boucle, affiche ce qu'il tape, et arrête quand il tape "exit"
"""


# ─── SOLUTION 3 (mini chatbot) ────────────────────────────────────────────────

def mini_chatbot():
    print("Chatbot démarré. Tape 'exit' pour quitter.\n")
    while True:
        user_input = input("Toi : ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("À bientôt !")
            break
        print(f"Bot : J'ai reçu → '{user_input}'\n")

# Décommente pour tester :
# mini_chatbot()
