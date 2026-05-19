"""
MODULE 1 — Leçon 4 : Gestion des erreurs
==========================================
Les agents IA appellent des APIs, lisent des fichiers, exécutent du code externe.
Tout peut échouer. Savoir gérer les erreurs = code robuste.
"""

# ─── TRY / EXCEPT DE BASE ─────────────────────────────────────────────────────

try:
    resultat = 10 / 0          # ZeroDivisionError
except ZeroDivisionError:
    print("Erreur : division par zéro !")

# Capturer l'erreur pour l'afficher
try:
    nombre = int("abc")        # ValueError
except ValueError as e:
    print(f"Erreur de conversion : {e}")


# ─── PLUSIEURS TYPES D'ERREURS ────────────────────────────────────────────────

def lire_config(data: dict, cle: str) -> str:
    try:
        valeur = data[cle]          # KeyError si la clé n'existe pas
        return str(valeur)          # ValueError si conversion impossible
    except KeyError:
        print(f"Clé '{cle}' introuvable dans la config")
        return ""
    except (TypeError, ValueError) as e:
        print(f"Valeur invalide pour '{cle}': {e}")
        return ""

config = {"modele": "claude-sonnet", "tokens": 4096}
print(lire_config(config, "modele"))    # "claude-sonnet"
print(lire_config(config, "timeout"))  # clé absente


# ─── FINALLY — s'exécute TOUJOURS ─────────────────────────────────────────────
# Utilise finally pour libérer des ressources (fermer un fichier, une connexion...)

def appeler_api_simulation(url: str) -> str:
    connexion_ouverte = False
    try:
        print(f"Connexion à {url}...")
        connexion_ouverte = True
        if "invalide" in url:
            raise ConnectionError("URL invalide")
        return "Réponse de l'API"
    except ConnectionError as e:
        print(f"Erreur réseau : {e}")
        return ""
    finally:
        if connexion_ouverte:
            print("Connexion fermée.")     # s'exécute même en cas d'erreur

print(appeler_api_simulation("https://api.openrouter.ai"))
print(appeler_api_simulation("https://url-invalide"))


# ─── EXCEPTIONS PERSONNALISÉES ────────────────────────────────────────────────
# Pour les agents, crée tes propres exceptions — c'est plus clair que ValueError

class LLMError(Exception):
    """Erreur générique lors d'un appel LLM."""
    pass

class RateLimitError(LLMError):
    """Trop de requêtes envoyées à l'API."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit dépassé. Réessaie dans {retry_after}s.")

class ContextTooLongError(LLMError):
    """Le contexte dépasse la fenêtre maximale du modèle."""
    pass


def simuler_appel_llm(tokens: int) -> str:
    if tokens > 200_000:
        raise ContextTooLongError(f"Contexte trop long : {tokens} tokens")
    if tokens < 0:
        raise LLMError("Le nombre de tokens ne peut pas être négatif")
    return f"Réponse générée ({tokens} tokens)"

try:
    print(simuler_appel_llm(1000))
    print(simuler_appel_llm(300_000))   # lèvera une erreur
except ContextTooLongError as e:
    print(f"Contexte dépassé : {e}")
except LLMError as e:
    print(f"Erreur LLM générique : {e}")


# ─── PATTERN RETRY — réessayer en cas d'échec ─────────────────────────────────
# Essentiel pour les appels API qui peuvent échouer temporairement

import time

def avec_retry(fonction, max_tentatives: int = 3, delai: float = 1.0):
    """Réessaie une fonction en cas d'échec."""
    for tentative in range(1, max_tentatives + 1):
        try:
            return fonction()
        except Exception as e:
            print(f"Tentative {tentative}/{max_tentatives} échouée : {e}")
            if tentative < max_tentatives:
                time.sleep(delai)
    raise RuntimeError(f"Échec après {max_tentatives} tentatives")

# Simulation
compteur = {"val": 0}

def operation_instable():
    compteur["val"] += 1
    if compteur["val"] < 3:
        raise ConnectionError("Serveur temporairement indisponible")
    return "Succès !"

try:
    resultat = avec_retry(operation_instable, max_tentatives=3, delai=0.1)
    print(resultat)
except RuntimeError as e:
    print(e)


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Crée une exception 'CleAPIManquanteError' qui hérite de ValueError.
   Lève-la si os.getenv("OPENROUTER_API_KEY") retourne None.

2. Écris une fonction 'charger_historique(fichier)' qui :
   - Retourne [] si le fichier n'existe pas (FileNotFoundError)
   - Retourne [] si le JSON est invalide (json.JSONDecodeError)
   - Retourne la liste des messages sinon

3. Ajoute un finally qui affiche "Chargement terminé" dans tous les cas.
"""
