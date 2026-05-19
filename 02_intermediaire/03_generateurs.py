"""
MODULE 2 — Leçon 3 : Générateurs & Itérateurs
===============================================
Les générateurs sont essentiels pour le streaming LLM :
au lieu d'attendre toute la réponse, tu traites chaque token
au fur et à mesure qu'il arrive.
"""

# ─── YIELD — retourner une valeur sans terminer la fonction ───────────────────
# Une fonction avec yield est un "générateur" — elle ne s'exécute pas
# immédiatement, elle produit des valeurs une par une à la demande.

def compter_jusqua(n: int):
    """Génère les entiers de 0 à n-1 sans tous les stocker en mémoire."""
    for i in range(n):
        yield i         # pause ici, retourne i, reprend au prochain next()

gen = compter_jusqua(5)
print(next(gen))    # 0
print(next(gen))    # 1
print(next(gen))    # 2

for valeur in compter_jusqua(3):    # itérer sur un générateur
    print(valeur)

# Différence cruciale avec une liste :
# list(range(1_000_000)) → alloue 8 MB en mémoire d'un coup
# compter_jusqua(1_000_000) → utilise quelques octets, calcule à la demande


# ─── STREAMING LLM — le vrai cas d'usage ──────────────────────────────────────
# L'API OpenRouter/OpenAI peut streamer les tokens un par un.
# Voici comment simuler et consommer un stream.

def simuler_stream_llm(texte: str):
    """Simule le streaming token par token d'un LLM."""
    mots = texte.split()
    for mot in mots:
        yield mot + " "


def afficher_stream(generateur):
    """Affiche les tokens au fur et à mesure (comme ChatGPT)."""
    reponse_complete = ""
    for token in generateur:
        print(token, end="", flush=True)    # flush=True force l'affichage immédiat
        reponse_complete += token
    print()  # saut de ligne final
    return reponse_complete.strip()


stream = simuler_stream_llm("Les agents IA peuvent appeler des outils externes.")
reponse = afficher_stream(stream)
print(f"Réponse complète : {reponse!r}")


# ─── GÉNÉRATEUR AVEC ÉTAT ─────────────────────────────────────────────────────

def generateur_messages(historique: list[dict]):
    """Parcourt un historique et produit des résumés formatés."""
    for i, msg in enumerate(historique, start=1):
        role = msg["role"].upper()
        contenu = msg["content"]
        extrait = contenu[:60] + "..." if len(contenu) > 60 else contenu
        yield f"[{i}] {role}: {extrait}"


historique = [
    {"role": "user",      "content": "Qu'est-ce qu'un agent IA ?"},
    {"role": "assistant", "content": "Un agent IA est un système autonome capable de percevoir son environnement, raisonner, et agir."},
    {"role": "user",      "content": "Donne-moi un exemple concret."},
]

for ligne in generateur_messages(historique):
    print(ligne)


# ─── COMPREHENSIONS AVANCÉES ──────────────────────────────────────────────────

# List comprehension — déjà vue
longueurs = [len(m["content"]) for m in historique]
print(longueurs)

# Dict comprehension — créer un dict en une ligne
index_messages = {i: m["role"] for i, m in enumerate(historique)}
print(index_messages)   # {0: 'user', 1: 'assistant', 2: 'user'}

# Set comprehension — valeurs uniques
roles_uniques = {m["role"] for m in historique}
print(roles_uniques)    # {'user', 'assistant'}

# Generator expression — comme list comprehension mais sans créer la liste
# Idéal pour les calculs one-shot sur de grandes collections
total_chars = sum(len(m["content"]) for m in historique)
print(f"Total caractères : {total_chars}")

messages_longs = list(
    m["content"] for m in historique if len(m["content"]) > 50
)
print(f"Messages longs : {len(messages_longs)}")


# ─── YIELD FROM — déléguer à un sous-générateur ───────────────────────────────

def messages_utilisateur(historique: list[dict]):
    for msg in historique:
        if msg["role"] == "user":
            yield msg["content"]

def messages_assistant(historique: list[dict]):
    for msg in historique:
        if msg["role"] == "assistant":
            yield msg["content"]

def tous_les_messages(historique: list[dict]):
    """Génère d'abord les messages user, puis assistant."""
    yield from messages_utilisateur(historique)     # délègue
    yield from messages_assistant(historique)


for msg in tous_les_messages(historique):
    print(f"  → {msg[:50]}")


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Écris un générateur 'chunker(texte, taille)' qui découpe un texte
   en morceaux de 'taille' caractères. Utile pour les gros documents.

2. Écris une fonction 'consommer_stream_openrouter(client, messages)'
   qui appelle l'API avec stream=True et yield chaque token reçu.
   Signature de l'API : client.chat.completions.create(..., stream=True)
   Chaque chunk a : chunk.choices[0].delta.content

3. Utilise une dict comprehension pour créer un index
   {role: [liste de messages]} depuis un historique.
"""


# ─── SOLUTION 1 ───────────────────────────────────────────────────────────────

def chunker(texte: str, taille: int):
    for debut in range(0, len(texte), taille):
        yield texte[debut:debut + taille]

texte_long = "Les agents IA sont des systèmes autonomes capables de raisonner."
for chunk in chunker(texte_long, 20):
    print(repr(chunk))


# ─── SOLUTION 3 ───────────────────────────────────────────────────────────────

from collections import defaultdict

def indexer_par_role(historique: list[dict]) -> dict[str, list[str]]:
    index = defaultdict(list)
    for msg in historique:
        index[msg["role"]].append(msg["content"])
    return dict(index)

print(indexer_par_role(historique))
