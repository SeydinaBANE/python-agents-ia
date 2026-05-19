"""
MODULE 1 — Leçon 3 : Fonctions
================================
Les fonctions sont le cœur du code réutilisable.
Dans les agents IA, chaque "tool" est une fonction Python.
"""

# ─── FONCTION DE BASE ─────────────────────────────────────────────────────────

def saluer(nom: str) -> str:
    """Retourne un message de salutation."""
    return f"Bonjour {nom} !"

print(saluer("Seydina"))


# ─── PARAMÈTRES PAR DÉFAUT ────────────────────────────────────────────────────

def creer_message(contenu: str, role: str = "user") -> dict:
    return {"role": role, "content": contenu}

print(creer_message("Bonjour !"))                       # role="user" par défaut
print(creer_message("Tu es un assistant.", role="system"))


# ─── *args — nombre variable d'arguments positionnels ─────────────────────────
# Utile quand tu ne sais pas combien d'arguments seront passés

def concatener(*morceaux: str) -> str:
    return " ".join(morceaux)

print(concatener("Les", "agents", "IA", "sont", "puissants"))


# ─── **kwargs — arguments nommés dynamiques ───────────────────────────────────
# Très utilisé pour passer des options à une API

def appeler_llm(prompt: str, **options) -> dict:
    parametres = {
        "model": "claude-sonnet",
        "max_tokens": 1024,
        "prompt": prompt,
    }
    parametres.update(options)     # les kwargs écrasent/ajoutent
    return parametres

print(appeler_llm("Dis bonjour", temperature=0.7, max_tokens=500))


# ─── FONCTIONS QUI RETOURNENT PLUSIEURS VALEURS ───────────────────────────────
# Python retourne un tuple, qu'on "dépaquète" directement

def analyser_message(message: str) -> tuple[int, bool, str]:
    longueur = len(message)
    est_question = message.endswith("?")
    premier_mot = message.split()[0] if message else ""
    return longueur, est_question, premier_mot

longueur, est_question, premier_mot = analyser_message("Comment vas-tu ?")
print(f"Longueur: {longueur}, Question: {est_question}, Premier mot: {premier_mot}")


# ─── LAMBDA — fonction anonyme sur une ligne ──────────────────────────────────
# Utile pour les transformations rapides, les tris, les callbacks

doubler = lambda x: x * 2
print(doubler(5))           # 10

# Trier une liste de dicts par une clé
agents = [
    {"nom": "Atlas", "score": 92},
    {"nom": "Orion", "score": 78},
    {"nom": "Nova",  "score": 95},
]
agents_tries = sorted(agents, key=lambda a: a["score"], reverse=True)
for a in agents_tries:
    print(f"{a['nom']}: {a['score']}")


# ─── FONCTIONS COMME VALEURS ──────────────────────────────────────────────────
# En Python, les fonctions sont des objets — on peut les passer en argument

def formater_majuscule(texte: str) -> str:
    return texte.upper()

def formater_titre(texte: str) -> str:
    return texte.title()

def appliquer_format(message: str, formateur) -> str:
    return formateur(message)

print(appliquer_format("bonjour monde", formater_majuscule))
print(appliquer_format("bonjour monde", formater_titre))

# C'est exactement comme ça que fonctionnent les "tools" dans les agents :
# un registre de fonctions qu'on appelle dynamiquement.


# ─── PORTÉE DES VARIABLES (scope) ─────────────────────────────────────────────

compteur_global = 0

def incrementer():
    global compteur_global      # sans 'global', Python crée une variable locale
    compteur_global += 1

incrementer()
incrementer()
print(compteur_global)          # 2


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Écris une fonction 'construire_historique' qui prend *messages (chaque message
   est une string) et retourne une liste de dicts {"role": "user", "content": msg}

2. Écris une fonction 'compter_tokens_approx' qui prend un texte et estime
   le nombre de tokens (approximation : 1 token ≈ 4 caractères)

3. Écris une lambda qui extrait le contenu d'un dict message :
   message = {"role": "user", "content": "Bonjour"}
"""


# ─── SOLUTIONS ────────────────────────────────────────────────────────────────

def construire_historique(*messages: str) -> list[dict]:
    return [{"role": "user", "content": msg} for msg in messages]

print(construire_historique("Bonjour", "Comment ça va ?", "Parle-moi des agents IA"))


def compter_tokens_approx(texte: str) -> int:
    return len(texte) // 4

print(compter_tokens_approx("Les agents IA sont des systèmes autonomes."))  # ≈ 10 tokens


extraire_contenu = lambda msg: msg["content"]
message = {"role": "user", "content": "Bonjour !"}
print(extraire_contenu(message))    # "Bonjour !"
