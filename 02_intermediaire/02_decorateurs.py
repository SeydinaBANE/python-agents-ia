"""
MODULE 2 — Leçon 2 : Décorateurs
==================================
Les décorateurs transforment des fonctions.
Dans les agents IA, ils servent à enregistrer des tools,
mesurer les temps de réponse, valider les inputs, loguer les appels.
"""

import time
import functools
from typing import Callable

# ─── @PROPERTY — attribut calculé ────────────────────────────────────────────

class Agent:
    def __init__(self, nom: str, tokens_utilises: int = 0, budget_tokens: int = 10_000):
        self.nom = nom
        self._tokens_utilises = tokens_utilises
        self._budget_tokens = budget_tokens

    @property
    def tokens_utilises(self) -> int:
        """Lecture : agent.tokens_utilises"""
        return self._tokens_utilises

    @tokens_utilises.setter
    def tokens_utilises(self, valeur: int) -> None:
        """Écriture avec validation : agent.tokens_utilises = 500"""
        if valeur < 0:
            raise ValueError("Les tokens ne peuvent pas être négatifs")
        self._tokens_utilises = valeur

    @property
    def budget_restant(self) -> int:
        """Attribut calculé — pas stocké, calculé à la demande"""
        return self._budget_tokens - self._tokens_utilises

    @property
    def est_actif(self) -> bool:
        return self.budget_restant > 0


a = Agent("Atlas", budget_tokens=1000)
a.tokens_utilises = 300
print(a.budget_restant)     # 700
print(a.est_actif)          # True
a.tokens_utilises = 1100
print(a.est_actif)          # False


# ─── DÉCORATEUR CUSTOM — fonction qui enveloppe une fonction ──────────────────
# Anatomie d'un décorateur :
#   1. Prend une fonction en entrée
#   2. Retourne une nouvelle fonction (le "wrapper")
#   3. Le wrapper fait quelque chose avant/après l'appel original

def mesurer_temps(fonction: Callable) -> Callable:
    """Affiche le temps d'exécution d'une fonction."""
    @functools.wraps(fonction)   # conserve le nom et la docstring de l'originale
    def wrapper(*args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        duree = time.perf_counter() - debut
        print(f"[{fonction.__name__}] durée : {duree:.3f}s")
        return resultat
    return wrapper


@mesurer_temps
def appel_llm_simule(prompt: str) -> str:
    time.sleep(0.1)     # simule une latence réseau
    return f"Réponse à : {prompt}"

print(appel_llm_simule("Bonjour"))


# ─── DÉCORATEUR AVEC PARAMÈTRES ───────────────────────────────────────────────
# Il faut une couche d'enveloppe supplémentaire

def retry(max_tentatives: int = 3, delai: float = 0.5):
    """Réessaie la fonction en cas d'exception."""
    def decorateur(fonction: Callable) -> Callable:
        @functools.wraps(fonction)
        def wrapper(*args, **kwargs):
            for tentative in range(1, max_tentatives + 1):
                try:
                    return fonction(*args, **kwargs)
                except Exception as e:
                    print(f"  Tentative {tentative}/{max_tentatives} échouée : {e}")
                    if tentative < max_tentatives:
                        time.sleep(delai)
            raise RuntimeError(f"Échec après {max_tentatives} tentatives")
        return wrapper
    return decorateur


compteur = {"val": 0}

@retry(max_tentatives=3, delai=0.05)
def operation_instable() -> str:
    compteur["val"] += 1
    if compteur["val"] < 3:
        raise ConnectionError("Serveur indisponible")
    return "Succès !"

print(operation_instable())


# ─── DÉCORATEUR TOOL REGISTRY — cœur des agents IA ───────────────────────────
# C'est exactement comme ça que LangChain, CrewAI etc. enregistrent les outils

class RegistreOutils:
    """Registre global des outils disponibles pour l'agent."""

    def __init__(self):
        self._outils: dict[str, dict] = {}

    def tool(self, description: str):
        """Décorateur qui enregistre une fonction comme outil d'agent."""
        def decorateur(fonction: Callable) -> Callable:
            self._outils[fonction.__name__] = {
                "fonction": fonction,
                "description": description,
                "nom": fonction.__name__,
            }
            return fonction
        return decorateur

    def executer(self, nom: str, **kwargs):
        if nom not in self._outils:
            raise KeyError(f"Outil inconnu : '{nom}'")
        return self._outils[nom]["fonction"](**kwargs)

    def lister(self) -> list[dict]:
        return [
            {"nom": o["nom"], "description": o["description"]}
            for o in self._outils.values()
        ]


# Utilisation — exactement le pattern des vrais frameworks d'agents
outils = RegistreOutils()

@outils.tool("Calcule la somme de deux nombres")
def additionner(a: float, b: float) -> float:
    return a + b

@outils.tool("Convertit un texte en majuscules")
def mettre_en_majuscule(texte: str) -> str:
    return texte.upper()

@outils.tool("Retourne la météo simulée d'une ville")
def get_meteo(ville: str) -> str:
    meteos = {"Paris": "nuageux 18°C", "Dakar": "ensoleillé 32°C", "Lyon": "pluvieux 14°C"}
    return meteos.get(ville, f"données indisponibles pour {ville}")


print(outils.lister())
print(outils.executer("additionner", a=3.5, b=2.1))
print(outils.executer("get_meteo", ville="Dakar"))


# ─── @STATICMETHOD et @CLASSMETHOD (rappel du module POO) ────────────────────

class Config:
    _instance = None    # pattern Singleton — une seule instance possible

    def __init__(self, modele: str):
        self.modele = modele

    @classmethod
    def get_instance(cls) -> "Config":
        """Retourne toujours la même instance (Singleton)."""
        if cls._instance is None:
            cls._instance = cls("anthropic/claude-sonnet-4-5")
        return cls._instance

    @staticmethod
    def modeles_disponibles() -> list[str]:
        return [
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-sonnet-4-5",
            "openai/gpt-4o",
            "mistralai/mistral-large",
        ]


c1 = Config.get_instance()
c2 = Config.get_instance()
print(c1 is c2)                         # True — même objet
print(Config.modeles_disponibles())


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée un décorateur 'logger' qui affiche le nom de la fonction,
   ses arguments, et sa valeur de retour à chaque appel.

2. Crée un décorateur 'valider_prompt' qui lève ValueError si
   le premier argument (le prompt) est vide ou None.

3. Ajoute au RegistreOutils une méthode 'schema_openai()' qui retourne
   les outils au format attendu par l'API OpenAI/OpenRouter :
   [{"type": "function", "function": {"name": ..., "description": ...}}]
"""
