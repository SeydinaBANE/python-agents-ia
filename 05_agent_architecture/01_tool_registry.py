"""
MODULE 5 — Leçon 1 : Tool Registry
=====================================
Un vrai registre d'outils génère le schéma JSON automatiquement
depuis les type hints et docstrings Python. Plus besoin d'écrire
le JSON Schema à la main comme au Module 4.

C'est exactement ce que font LangChain, CrewAI, et le SDK Anthropic.
"""

import inspect
import json
from typing import Any, Callable, get_type_hints
from pydantic import BaseModel, Field, create_model
from rich.console import Console

console = Console()


# ─── 1. DÉCORATEUR @tool ──────────────────────────────────────────────────────

def _python_type_to_json_schema(annotation) -> dict:
    """Convertit un type Python en JSON Schema."""
    mapping = {
        str:   {"type": "string"},
        int:   {"type": "integer"},
        float: {"type": "number"},
        bool:  {"type": "boolean"},
        list:  {"type": "array"},
        dict:  {"type": "object"},
    }
    return mapping.get(annotation, {"type": "string"})


class ToolRegistry:
    """
    Registre d'outils qui génère automatiquement le JSON Schema
    depuis les signatures et docstrings des fonctions Python.
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def tool(self, description: str | None = None):
        """
        Décorateur qui enregistre une fonction comme outil d'agent.
        Le schéma JSON est généré automatiquement depuis les type hints.

        Usage :
            @registry.tool("Calcule la somme de deux nombres")
            def additionner(a: int, b: int) -> int:
                return a + b
        """
        def decorateur(fn: Callable) -> Callable:
            desc = description or inspect.getdoc(fn) or fn.__name__
            sig  = inspect.signature(fn)
            hints = get_type_hints(fn)

            properties = {}
            required   = []

            for nom, param in sig.parameters.items():
                schema = _python_type_to_json_schema(hints.get(nom, str))

                # Récupère la valeur par défaut si elle existe
                if param.default is inspect.Parameter.empty:
                    required.append(nom)
                else:
                    schema["default"] = param.default

                # Essaie de lire la description depuis le commentaire inline
                properties[nom] = schema

            self._tools[fn.__name__] = {
                "fn":     fn,
                "schema": {
                    "type": "function",
                    "function": {
                        "name":        fn.__name__,
                        "description": desc,
                        "parameters": {
                            "type":       "object",
                            "properties": properties,
                            "required":   required,
                        },
                    },
                },
            }
            return fn
        return decorateur

    def executer(self, nom: str, **kwargs) -> Any:
        if nom not in self._tools:
            return {"erreur": f"Tool '{nom}' inconnu. Disponibles : {self.noms()}"}
        try:
            return self._tools[nom]["fn"](**kwargs)
        except Exception as e:
            return {"erreur": str(e)}

    def schemas(self) -> list[dict]:
        """Retourne la liste des schémas au format OpenAI/OpenRouter."""
        return [t["schema"] for t in self._tools.values()]

    def noms(self) -> list[str]:
        return list(self._tools.keys())

    def afficher(self) -> None:
        from rich.table import Table
        table = Table(title=f"Outils enregistrés ({len(self._tools)})")
        table.add_column("Nom",         style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Paramètres",  style="yellow")
        for nom, data in self._tools.items():
            schema = data["schema"]["function"]
            params = list(schema["parameters"]["properties"].keys())
            table.add_row(nom, schema["description"][:60], str(params))
        console.print(table)


# ─── 2. DÉFINIR DES OUTILS RÉELS ─────────────────────────────────────────────

registry = ToolRegistry()

@registry.tool("Retourne la météo actuelle d'une ville (simulation)")
def get_meteo(ville: str, unite: str = "celsius") -> dict:
    donnees = {
        "Paris":  {"temp": 18, "cond": "nuageux"},
        "Dakar":  {"temp": 32, "cond": "ensoleillé"},
        "Lyon":   {"temp": 15, "cond": "pluvieux"},
        "Tokyo":  {"temp": 22, "cond": "brumeux"},
    }
    info = donnees.get(ville, {"temp": 20, "cond": "inconnu"})
    temp = info["temp"]
    if unite == "fahrenheit":
        temp = round(temp * 9/5 + 32, 1)
    return {"ville": ville, "temperature": f"{temp}°", "condition": info["cond"]}


@registry.tool("Évalue une expression mathématique")
def calculer(expression: str) -> dict:
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Expression non autorisée"}
        return {"resultat": eval(expression)}  # noqa: S307
    except Exception as e:
        return {"erreur": str(e)}


@registry.tool("Compte le nombre de mots dans un texte")
def compter_mots(texte: str) -> dict:
    mots = texte.split()
    return {"nb_mots": len(mots), "nb_chars": len(texte)}


@registry.tool("Convertit une température entre Celsius et Fahrenheit")
def convertir_temp(valeur: float, de: str, vers: str) -> dict:
    if de == "celsius" and vers == "fahrenheit":
        return {"resultat": round(valeur * 9/5 + 32, 2), "unite": "°F"}
    if de == "fahrenheit" and vers == "celsius":
        return {"resultat": round((valeur - 32) * 5/9, 2), "unite": "°C"}
    return {"erreur": f"Conversion {de} → {vers} non supportée"}


@registry.tool("Génère un résumé fictif d'une URL (simulation)")
def lire_url(url: str, max_mots: int = 100) -> dict:
    return {
        "url":     url,
        "titre":   f"Page simulée : {url.split('/')[-1] or 'accueil'}",
        "contenu": f"Contenu fictif de {url} (limité à {max_mots} mots)",
    }


# ─── 3. INSPECTER LE REGISTRE ────────────────────────────────────────────────

registry.afficher()

# Le JSON Schema généré automatiquement
print("\nSchéma JSON de 'calculer' :")
print(json.dumps(registry._tools["calculer"]["schema"], indent=2, ensure_ascii=False))


# ─── 4. EXECUTER DES OUTILS ───────────────────────────────────────────────────

print("\nExécutions :")
print(registry.executer("get_meteo", ville="Dakar"))
print(registry.executer("calculer", expression="(15 + 7) * 3"))
print(registry.executer("compter_mots", texte="Les agents IA sont des systèmes autonomes"))
print(registry.executer("convertir_temp", valeur=100.0, de="celsius", vers="fahrenheit"))
print(registry.executer("outil_inconnu"))       # erreur propre


# ─── 5. REGISTRE AVEC PYDANTIC — schéma riche ────────────────────────────────
# Pour les tools avec des paramètres complexes, Pydantic génère
# un schéma JSON Schema complet avec descriptions par champ.

class RechercheInput(BaseModel):
    """Paramètres pour la recherche web."""
    query:     str = Field(description="Terme de recherche")
    max_results: int = Field(default=5, ge=1, le=20, description="Nombre max de résultats")
    langue:    str = Field(default="fr", description="Code langue ISO")

class RechercheOutput(BaseModel):
    resultats: list[str]
    total:     int
    duree_ms:  float

def recherche_web(query: str, max_results: int = 5, langue: str = "fr") -> dict:
    """Effectue une recherche web (simulation)."""
    return RechercheOutput(
        resultats=[f"Résultat {i+1} pour '{query}'" for i in range(max_results)],
        total=max_results,
        duree_ms=42.5,
    ).model_dump()

# Schéma Pydantic → tool OpenAI
schema_pydantic = {
    "type": "function",
    "function": {
        "name":        "recherche_web",
        "description": "Effectue une recherche web",
        "parameters":  RechercheInput.model_json_schema(),
    },
}
print("\nSchéma Pydantic :")
print(json.dumps(schema_pydantic, indent=2, ensure_ascii=False))


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Ajoute un tool 'traduire(texte, langue_source, langue_cible)' au registry.
   Simule la traduction en retournant le texte avec "[traduit en {langue_cible}]".

2. Modifie ToolRegistry pour ajouter une méthode 'get_schema(nom)'
   qui retourne le schéma d'un outil précis.

3. Ajoute une validation dans executer() :
   si un paramètre required manque dans kwargs, retourne une erreur claire
   avant même d'appeler la fonction.
"""
