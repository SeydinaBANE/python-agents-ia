"""
MODULE 7 — Leçon 2 : Tests
============================
Tester un agent IA c'est différent de tester du code classique :
  - Les LLMs sont non déterministes → on mocke les appels
  - Les tools sont déterministes    → on les teste directement
  - L'API FastAPI                   → on utilise TestClient

Lance : pytest 07_production/02_tests.py -v
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ─── 1. TESTER LES TOOLS DIRECTEMENT ────────────────────────────────────────
# Les tools sont des fonctions pures → tests simples, pas de mock

def get_meteo(ville: str) -> dict:
    meteos = {
        "Paris": {"temp": 18, "condition": "nuageux"},
        "Dakar": {"temp": 32, "condition": "ensoleillé"},
    }
    return meteos.get(ville, {"temp": 20, "condition": "inconnu"})

def calculer(expression: str) -> dict:
    try:
        autorise = set("0123456789+-*/()., ")
        if not all(c in autorise for c in expression):
            return {"erreur": "Non autorisé"}
        return {"resultat": eval(expression)}  # noqa: S307
    except Exception as e:
        return {"erreur": str(e)}

def compter_mots(texte: str) -> dict:
    mots = texte.split()
    return {"nb_mots": len(mots), "nb_chars": len(texte)}


class TestTools:
    def test_meteo_ville_connue(self):
        resultat = get_meteo("Paris")
        assert resultat["temp"] == 18
        assert resultat["condition"] == "nuageux"

    def test_meteo_ville_inconnue(self):
        resultat = get_meteo("Tombouctou")
        assert resultat["temp"] == 20
        assert resultat["condition"] == "inconnu"

    def test_calculer_addition(self):
        assert calculer("2 + 3")["resultat"] == 5

    def test_calculer_multiplication(self):
        assert calculer("15 * 8")["resultat"] == 120

    def test_calculer_expression_complexe(self):
        assert calculer("(10 + 5) * 2")["resultat"] == 30

    def test_calculer_expression_interdite(self):
        resultat = calculer("__import__('os').system('ls')")
        assert "erreur" in resultat

    def test_calculer_division_par_zero(self):
        resultat = calculer("1 / 0")
        assert "erreur" in resultat

    def test_compter_mots(self):
        r = compter_mots("Bonjour monde Python")
        assert r["nb_mots"] == 3
        assert r["nb_chars"] == 20

    def test_compter_mots_vide(self):
        r = compter_mots("")
        assert r["nb_mots"] == 0


# ─── 2. MOCKER LES APPELS LLM ────────────────────────────────────────────────
# Les LLMs coûtent de l'argent et sont lents → on les mocke dans les tests

class FakeChoix:
    def __init__(self, contenu: str, finish_reason: str = "stop"):
        self.message       = MagicMock(content=contenu, tool_calls=None)
        self.finish_reason = finish_reason
        self.delta         = MagicMock(content=contenu)

class FakeUsage:
    prompt_tokens     = 10
    completion_tokens = 20
    total_tokens      = 30

class FakeResponse:
    def __init__(self, contenu: str):
        self.id      = "fake-id-123"
        self.model   = "fake-model"
        self.choices = [FakeChoix(contenu)]
        self.usage   = FakeUsage()


class TestAgentAvecMock:
    @pytest.mark.asyncio
    async def test_appel_llm_mocke(self):
        """Vérifie que l'agent appelle bien le LLM avec les bons paramètres."""

        fake_response = FakeResponse("Bonjour ! Je suis un agent IA.")

        with patch("openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = AsyncMock(return_value=fake_response)

            # Simule un appel
            client = MockClient()
            response = await client.chat.completions.create(
                model="anthropic/claude-haiku-4-5",
                messages=[{"role": "user", "content": "Bonjour"}],
                max_tokens=100,
            )

            assert response.choices[0].message.content == "Bonjour ! Je suis un agent IA."
            assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_tool_call_mocke(self):
        """Vérifie le parsing des tool_calls dans la réponse LLM."""

        tool_call = MagicMock()
        tool_call.function.name      = "get_meteo"
        tool_call.function.arguments = json.dumps({"ville": "Paris"})
        tool_call.id                 = "call_abc123"

        choix = MagicMock()
        choix.finish_reason          = "tool_calls"
        choix.message.tool_calls     = [tool_call]
        choix.message.content        = None

        response  = MagicMock()
        response.choices = [choix]

        # Simuler le parsing
        assert response.choices[0].finish_reason == "tool_calls"
        assert len(response.choices[0].message.tool_calls) == 1

        tc   = response.choices[0].message.tool_calls[0]
        nom  = tc.function.name
        args = json.loads(tc.function.arguments)

        assert nom == "get_meteo"
        assert args["ville"] == "Paris"


# ─── 3. TESTER L'API FASTAPI ──────────────────────────────────────────────────

def _charger_fastapi_app():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("fastapi_app", "07_production/01_fastapi.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app

@pytest.fixture
def client_api():
    """Crée un client de test FastAPI avec le LLM mocké."""
    app = _charger_fastapi_app()

    fake_response = FakeResponse("Voici ma réponse de test.")

    with patch("openai.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=fake_response)

        with TestClient(app) as c:
            yield c


class TestAPI:
    def test_health(self, client_api):
        response = client_api.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["statut"] == "ok"
        assert "modeles" in data

    def test_chat_basique(self, client_api):
        response = client_api.post("/chat", json={
            "contenu": "Bonjour !",
            "session_id": "test-session-1",
        })
        assert response.status_code == 200
        data = response.json()
        assert "reponse" in data
        assert "session_id" in data
        assert "tokens" in data

    def test_chat_session_persistee(self, client_api):
        session_id = "test-persistance"
        client_api.post("/chat", json={"contenu": "Premier message", "session_id": session_id})
        client_api.post("/chat", json={"contenu": "Deuxième message", "session_id": session_id})

        response = client_api.get(f"/historique/{session_id}")
        assert response.status_code == 200
        assert response.json()["nb_messages"] == 4  # 2 user + 2 assistant

    def test_session_inexistante(self, client_api):
        response = client_api.get("/historique/session-inexistante-xyz")
        assert response.status_code == 404

    def test_supprimer_session(self, client_api):
        session_id = "test-delete"
        client_api.post("/chat", json={"contenu": "Test", "session_id": session_id})
        response = client_api.delete(f"/historique/{session_id}")
        assert response.status_code == 200
        # Vérifier que la session n'existe plus
        response = client_api.get(f"/historique/{session_id}")
        assert response.status_code == 404

    def test_contenu_vide_rejete(self, client_api):
        response = client_api.post("/chat", json={"contenu": ""})
        assert response.status_code == 422   # validation error Pydantic


# ─── 4. TESTER LA MÉMOIRE ────────────────────────────────────────────────────

def _charger_memory():
    import importlib.util
    spec = importlib.util.spec_from_file_location("memory", "05_agent_architecture/02_memory.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MemoireCourte, mod.MemoireLongue

MemoireCourte, MemoireLongue = _charger_memory()


class TestMemoire:
    def test_memoire_courte_ajouter(self):
        mem = MemoireCourte("Tu es un assistant.")
        mem.ajouter("user", "Bonjour")
        mem.ajouter("assistant", "Bonjour !")
        assert mem.nb_messages == 2

    def test_memoire_courte_troncature(self):
        mem = MemoireCourte("System", max_messages=4)
        for i in range(10):
            mem.ajouter("user", f"msg {i}")
        msgs = mem.pour_api()
        # system + 4 messages max
        assert len(msgs) <= 5

    def test_memoire_courte_tokens_approx(self):
        mem = MemoireCourte("System prompt court")
        mem.ajouter("user", "a" * 400)   # ~100 tokens
        assert mem.tokens_approx > 0

    def test_memoire_longue_memoriser(self, tmp_path):
        chemin = str(tmp_path / "test_mem.json")
        mem    = MemoireLongue(chemin)
        s      = mem.memoriser("L'utilisateur s'appelle Seydina", categorie="utilisateur")
        assert s.contenu == "L'utilisateur s'appelle Seydina"
        assert len(mem._souvenirs) == 1

    def test_memoire_longue_persistence(self, tmp_path):
        chemin = str(tmp_path / "test_persist.json")
        mem1   = MemoireLongue(chemin)
        mem1.memoriser("Fait important", importance=5)

        mem2   = MemoireLongue(chemin)   # recharge depuis le fichier
        assert len(mem2._souvenirs) == 1
        assert mem2._souvenirs[0].importance == 5

    def test_memoire_longue_rechercher(self, tmp_path):
        chemin = str(tmp_path / "test_search.json")
        mem    = MemoireLongue(chemin)
        mem.memoriser("L'utilisateur aime Python")
        mem.memoriser("Il travaille sur un agent IA")
        mem.memoriser("Il préfère les réponses courtes")

        resultats = mem.rechercher("Python")
        assert len(resultats) >= 1
        assert "Python" in resultats[0].contenu


# ─── LANCEMENT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
