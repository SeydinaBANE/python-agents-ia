"""
MODULE 3 — Leçon 3 : Loguru
=============================
Loguru remplace le module logging standard avec une API bien plus simple.
Dans les agents IA, les logs sont essentiels pour déboguer les appels LLM,
tracer les décisions de l'agent, et mesurer les coûts en tokens.
"""

import sys
import time
from pathlib import Path
from loguru import logger

# ─── CONFIGURATION DE BASE ────────────────────────────────────────────────────
# Par défaut, loguru affiche dans stderr avec couleurs.
# On reconfigure pour contrôler le format et les destinations.

logger.remove()     # supprime le handler par défaut

# Handler console — coloré, pour le développement
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

# Handler fichier — tout sauvegarder pour analyser plus tard
Path("03_ecosysteme_ia/data").mkdir(exist_ok=True)
logger.add(
    "03_ecosysteme_ia/data/agent_{time:YYYY-MM-DD}.log",
    rotation="1 day",       # nouveau fichier chaque jour
    retention="7 days",     # garde 7 jours d'historique
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
)


# ─── LES NIVEAUX DE LOG ───────────────────────────────────────────────────────
# DEBUG    → infos détaillées pour déboguer (tokens, durées, payloads)
# INFO     → événements normaux (agent démarré, message reçu, tool appelé)
# WARNING  → situation anormale mais pas bloquante (rate limit proche, retry)
# ERROR    → erreur récupérée (appel API échoué, tool cassé)
# CRITICAL → erreur fatale (clé API manquante, crash complet)

logger.debug("Payload envoyé : {}", {"model": "claude-haiku", "tokens": 150})
logger.info("Agent démarré avec le modèle claude-haiku-4-5")
logger.warning("Budget tokens à 80% ({}/{})", 8000, 10000)
logger.error("Appel API échoué après 3 tentatives")
logger.critical("Clé API manquante — arrêt immédiat")


# ─── CONTEXTUALISER LES LOGS AVEC BIND ────────────────────────────────────────
# bind() ajoute des champs qui apparaissent dans tous les logs suivants
# Idéal pour tracer une session ou une conversation spécifique

def traiter_session(session_id: str, user_id: str):
    log = logger.bind(session=session_id, user=user_id)
    log.info("Session démarrée")
    log.debug("Chargement de l'historique")
    log.info("Réponse envoyée ({} tokens)", 245)
    log.info("Session terminée")

traiter_session("sess_abc123", "user_42")


# ─── DÉCORATEUR @logger.catch ─────────────────────────────────────────────────
# Capture automatiquement les exceptions non gérées et les logue

@logger.catch
def operation_risquee(x: int) -> int:
    return 10 // x      # ZeroDivisionError si x=0

operation_risquee(5)    # OK
operation_risquee(0)    # Loggué automatiquement avec traceback complet


# ─── LOGGER DANS UN AGENT ─────────────────────────────────────────────────────

class AgentLogger:
    """Exemple d'agent qui logue toutes ses actions."""

    def __init__(self, nom: str):
        self.nom = nom
        self.log = logger.bind(agent=nom)
        self.tokens_utilises = 0
        self.log.info("Agent '{}' initialisé", nom)

    def recevoir_message(self, contenu: str) -> None:
        self.log.info("Message reçu ({} chars)", len(contenu))
        self.log.debug("Contenu : {!r}", contenu[:100])

    def appeler_tool(self, nom_tool: str, **kwargs) -> None:
        self.log.info("Tool appelé : {} avec args={}", nom_tool, kwargs)
        debut = time.perf_counter()
        time.sleep(0.01)    # simule l'exécution
        duree = time.perf_counter() - debut
        self.log.debug("Tool '{}' terminé en {:.3f}s", nom_tool, duree)

    def appel_llm(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_utilises += tokens_in + tokens_out
        self.log.info(
            "Appel LLM : {} tokens in, {} tokens out (total: {})",
            tokens_in, tokens_out, self.tokens_utilises,
        )
        if self.tokens_utilises > 8_000:
            self.log.warning("Budget tokens à 80% ({}/10000)", self.tokens_utilises)

    def erreur(self, message: str, exc: Exception | None = None) -> None:
        if exc:
            self.log.exception("Erreur : {}", message)
        else:
            self.log.error("Erreur : {}", message)


agent = AgentLogger("Atlas")
agent.recevoir_message("Quel temps fait-il à Paris ?")
agent.appeler_tool("get_meteo", ville="Paris")
agent.appel_llm(tokens_in=150, tokens_out=80)
agent.appel_llm(tokens_in=200, tokens_out=120)
agent.appel_llm(tokens_in=400, tokens_out=200)   # → warning budget


# ─── MESURER LES PERFORMANCES ─────────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def log_duree(label: str):
    debut = time.perf_counter()
    logger.debug("Début : {}", label)
    try:
        yield
    finally:
        duree = time.perf_counter() - debut
        logger.info("{} terminé en {:.3f}s", label, duree)

with log_duree("appel LLM simulé"):
    time.sleep(0.05)


# ─── EXERCICES ────────────────────────────────────────────────────────────────
"""
1. Ajoute un handler loguru qui envoie les logs ERROR et CRITICAL
   dans un fichier séparé 'errors.log' (sans rotation).

2. Crée un décorateur 'log_appel' qui logue automatiquement :
   - Le nom de la fonction et ses arguments (au niveau DEBUG)
   - Sa valeur de retour (au niveau DEBUG)
   - Son temps d'exécution (au niveau INFO)
   - Toute exception levée (au niveau ERROR)

3. Modifie AgentLogger pour inclure un champ 'run_id' unique
   (uuid4) à chaque démarrage, visible dans tous les logs.
"""
