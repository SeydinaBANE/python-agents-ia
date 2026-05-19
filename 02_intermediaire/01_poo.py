"""
MODULE 2 — Leçon 1 : Programmation Orientée Objet (POO)
=========================================================
Les agents IA sont presque toujours modélisés comme des classes.
Un agent a un état (attributs) et des comportements (méthodes).
"""

# ─── CLASSE DE BASE ───────────────────────────────────────────────────────────

class Agent:
    """Représente un agent IA avec un nom et un historique."""

    # Attribut de classe — partagé par toutes les instances
    nb_agents_crees = 0

    def __init__(self, nom: str, modele: str = "claude-haiku"):
        # Attributs d'instance — propres à chaque objet
        self.nom = nom
        self.modele = modele
        self.historique: list[dict] = []
        self._actif = True                  # convention : _ = semi-privé
        Agent.nb_agents_crees += 1

    # ── Méthodes magiques ──────────────────────────────────────────────────────
    # Python les appelle automatiquement dans certains contextes

    def __str__(self) -> str:
        """Appelé par print() et str()"""
        return f"Agent(nom={self.nom}, modele={self.modele})"

    def __repr__(self) -> str:
        """Appelé dans le REPL et les listes — doit être non ambigu"""
        return f"Agent(nom={self.nom!r}, modele={self.modele!r})"

    def __len__(self) -> int:
        """Appelé par len() — ici : nombre de messages dans l'historique"""
        return len(self.historique)

    def __eq__(self, other: object) -> bool:
        """Appelé par == — deux agents sont égaux s'ils ont le même nom"""
        if not isinstance(other, Agent):
            return NotImplemented
        return self.nom == other.nom

    # ── Méthodes normales ──────────────────────────────────────────────────────

    def ajouter_message(self, role: str, contenu: str) -> None:
        self.historique.append({"role": role, "content": contenu})

    def effacer_historique(self) -> None:
        self.historique.clear()

    def resumer(self) -> str:
        return f"{self.nom} : {len(self)} messages, modèle={self.modele}"

    # ── Méthode de classe — alternative constructor ────────────────────────────
    # Utile pour créer un agent depuis un dict (ex: fichier de config JSON)

    @classmethod
    def depuis_dict(cls, data: dict) -> "Agent":
        return cls(nom=data["nom"], modele=data.get("modele", "claude-haiku"))

    # ── Méthode statique — fonction utilitaire liée à la classe ───────────────
    # Pas accès à self ni cls — c'est juste une fonction rangée dans la classe

    @staticmethod
    def valider_modele(modele: str) -> bool:
        modeles_valides = {"claude-haiku", "claude-sonnet", "gpt-4o", "mistral"}
        return modele in modeles_valides


# Tests
a1 = Agent("Atlas", "claude-sonnet")
a2 = Agent.depuis_dict({"nom": "Nova", "modele": "gpt-4o"})

print(a1)                           # Agent(nom=Atlas, modele=claude-sonnet)
print(repr(a2))                     # Agent(nom='Nova', modele='gpt-4o')
print(Agent.nb_agents_crees)        # 2

a1.ajouter_message("user", "Bonjour")
a1.ajouter_message("assistant", "Bonjour !")
print(len(a1))                      # 2
print(a1.resumer())

print(Agent.valider_modele("claude-sonnet"))    # True
print(Agent.valider_modele("llama-3"))          # False

a3 = Agent("Atlas")
print(a1 == a3)                     # True (même nom)
print(a1 == a2)                     # False


# ─── HÉRITAGE ─────────────────────────────────────────────────────────────────
# Spécialiser un agent en héritant de la classe de base

class AgentAvecOutils(Agent):
    """Agent capable d'utiliser des outils (tools)."""

    def __init__(self, nom: str, modele: str = "claude-sonnet"):
        super().__init__(nom, modele)       # appelle __init__ du parent
        self.outils: dict[str, callable] = {}

    def enregistrer_outil(self, nom: str, fonction: callable) -> None:
        self.outils[nom] = fonction

    def executer_outil(self, nom: str, **kwargs):
        if nom not in self.outils:
            raise KeyError(f"Outil '{nom}' inconnu. Disponibles : {list(self.outils)}")
        return self.outils[nom](**kwargs)

    def __str__(self) -> str:
        outils_liste = list(self.outils.keys())
        return f"AgentAvecOutils(nom={self.nom}, outils={outils_liste})"


class AgentMemoire(Agent):
    """Agent avec mémoire longue terme (résumé de sessions précédentes)."""

    def __init__(self, nom: str):
        super().__init__(nom)
        self.memoire_longue: list[str] = []

    def memoriser(self, fait: str) -> None:
        self.memoire_longue.append(fait)

    def contexte_systeme(self) -> str:
        if not self.memoire_longue:
            return "Tu es un assistant IA."
        faits = "\n".join(f"- {f}" for f in self.memoire_longue)
        return f"Tu es un assistant IA. Ce que tu sais sur l'utilisateur :\n{faits}"


# Tests héritage
agent_tools = AgentAvecOutils("Orion")
agent_tools.enregistrer_outil("calculer", lambda x, y: x + y)
agent_tools.enregistrer_outil("majuscule", lambda texte: texte.upper())

print(agent_tools.executer_outil("calculer", x=3, y=7))    # 10
print(agent_tools.executer_outil("majuscule", texte="hello"))  # HELLO
print(agent_tools)

agent_mem = AgentMemoire("Sage")
agent_mem.memoriser("L'utilisateur s'appelle Seydina")
agent_mem.memoriser("Il apprend Python pour faire des agents IA")
print(agent_mem.contexte_systeme())

# isinstance — vérifier la hiérarchie
print(isinstance(agent_tools, Agent))           # True — AgentAvecOutils EST un Agent
print(isinstance(agent_tools, AgentMemoire))    # False


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée une classe 'AgentRAG' qui hérite de Agent et ajoute :
   - Un attribut 'documents: list[str]' (liste de textes indexés)
   - Une méthode 'ajouter_document(texte)'
   - Une méthode 'rechercher(query)' qui retourne les documents
     contenant la query (insensible à la casse)

2. Ajoute __len__ qui retourne le nombre de documents (pas de messages)

3. Surcharge __str__ pour afficher aussi le nombre de documents
"""
