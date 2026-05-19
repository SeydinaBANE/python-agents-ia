"""
MODULE 2 — Leçon 6 : Type Hints
=================================
Les type hints documentent ton code ET permettent à Pylance/mypy
de détecter des bugs avant l'exécution. Dans les agents, ils
clarifient exactement ce qu'un tool attend et retourne.
"""

from typing import Optional, Union, Any
from collections.abc import Callable, Generator, AsyncGenerator
from typing import TypedDict, Protocol, runtime_checkable

# ─── TYPES SIMPLES ────────────────────────────────────────────────────────────

def saluer(nom: str) -> str:
    return f"Bonjour {nom}"

def calculer(a: int, b: int) -> float:
    return a / b

def rien_retourner(message: str) -> None:
    print(message)


# ─── COLLECTIONS TYPÉES ───────────────────────────────────────────────────────

def filtrer_messages(
    historique: list[dict[str, str]],
    role: str,
) -> list[dict[str, str]]:
    return [m for m in historique if m["role"] == role]

def compter_par_role(historique: list[dict]) -> dict[str, int]:
    compteur: dict[str, int] = {}
    for msg in historique:
        compteur[msg["role"]] = compteur.get(msg["role"], 0) + 1
    return compteur


# ─── OPTIONAL — peut être None ────────────────────────────────────────────────

def chercher_message(historique: list[dict], index: int) -> Optional[dict]:
    """Retourne None si l'index est hors limites."""
    if 0 <= index < len(historique):
        return historique[index]
    return None

# Optional[X] est un raccourci pour Union[X, None]
# En Python 3.10+ on peut écrire directement : dict | None
def chercher_v2(historique: list[dict], index: int) -> dict | None:
    if 0 <= index < len(historique):
        return historique[index]
    return None


# ─── UNION — plusieurs types possibles ───────────────────────────────────────

def parser_tokens(valeur: str | int | float) -> int:
    """Accepte '1024', 1024, ou 1024.0 et retourne toujours un int."""
    return int(valeur)

print(parser_tokens("2048"))    # 2048
print(parser_tokens(512))       # 512


# ─── TYPEDDICT — dict avec une structure connue ───────────────────────────────
# Beaucoup plus précis que dict[str, Any] pour les messages LLM

class Message(TypedDict):
    role: str
    content: str

class MessageAvecTimestamp(Message, total=False):
    # total=False → les clés héritées restent requises,
    # les nouvelles sont optionnelles
    timestamp: str
    tokens: int

def creer_message(role: str, contenu: str) -> Message:
    return {"role": role, "content": contenu}

historique: list[Message] = [
    creer_message("user", "Bonjour"),
    creer_message("assistant", "Bonjour !"),
]

# Pylance sait maintenant exactement quelles clés existent
print(historique[0]["content"])     # pas d'avertissement
# print(historique[0]["foo"])       # Pylance souligne en rouge ← c'est utile !


# ─── CALLABLE — passer des fonctions comme arguments ─────────────────────────

def appliquer(texte: str, transformation: Callable[[str], str]) -> str:
    return transformation(texte)

print(appliquer("bonjour", str.upper))
print(appliquer("MONDE", str.lower))


# ─── TYPE D'UN AGENT — exemple réaliste ──────────────────────────────────────

class ConfigAgent(TypedDict):
    nom: str
    modele: str
    max_tokens: int
    temperature: float
    outils: list[str]

class ResultatTool(TypedDict):
    nom_outil: str
    succes: bool
    resultat: Any
    erreur: str | None

def executer_tool(
    nom: str,
    registre: dict[str, Callable],
    **kwargs: Any,
) -> ResultatTool:
    if nom not in registre:
        return {"nom_outil": nom, "succes": False, "resultat": None, "erreur": f"Outil '{nom}' inconnu"}
    try:
        valeur = registre[nom](**kwargs)
        return {"nom_outil": nom, "succes": True, "resultat": valeur, "erreur": None}
    except Exception as e:
        return {"nom_outil": nom, "succes": False, "resultat": None, "erreur": str(e)}


registre = {
    "additionner": lambda a, b: a + b,
    "majuscule":   lambda texte: texte.upper(),
}

print(executer_tool("additionner", registre, a=3, b=4))
print(executer_tool("inexistant",  registre))


# ─── PROTOCOL — typage structurel ("duck typing" formalisé) ──────────────────
# Définit une interface sans héritage forcé

@runtime_checkable
class SupportsChat(Protocol):
    """Tout objet qui implémente cette interface peut être utilisé comme agent."""
    async def chat(self, message: str) -> str: ...
    def effacer_historique(self) -> None: ...


# ─── GÉNÉRATEURS ET ASYNC TYPÉS ───────────────────────────────────────────────

def generateur_chunks(texte: str, taille: int) -> Generator[str, None, None]:
    for i in range(0, len(texte), taille):
        yield texte[i:i + taille]

async def stream_tokens(prompt: str) -> AsyncGenerator[str, None]:
    mots = prompt.split()
    for mot in mots:
        yield mot + " "


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Définis un TypedDict 'AppelTool' qui représente un appel d'outil
   avec les champs : nom (str), arguments (dict[str, Any]), id (str)

2. Définis un TypedDict 'ReponseAPI' avec :
   id (str), modele (str), contenu (str), tokens_prompt (int),
   tokens_completion (int), et finish_reason (str | None)

3. Écris une fonction 'parser_reponse_openrouter(data: dict) -> ReponseAPI'
   qui extrait ces champs depuis la réponse brute de l'API.
"""
