"""
MODULE 7 — Leçon 3 : RAG (Retrieval-Augmented Generation)
===========================================================
RAG = donner à l'agent accès à une base de connaissances.
Au lieu d'halluciner, il cherche les faits dans tes documents.

Flux :
  1. Indexation  : découper les documents en chunks → embeddings → ChromaDB
  2. Retrieval   : question → embedding → chercher les chunks proches
  3. Generation  : injecter les chunks dans le prompt → LLM répond avec les faits

Lance : python 07_production/03_rag.py
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

Path("07_production/data").mkdir(parents=True, exist_ok=True)


# ─── 1. BASE DE CONNAISSANCES DE DÉMONSTRATION ───────────────────────────────

DOCUMENTS = [
    {
        "id":     "doc_1",
        "titre":  "Python asyncio",
        "contenu": "asyncio est la bibliothèque Python pour la programmation asynchrone. "
                   "Elle utilise une boucle d'événements pour exécuter des coroutines en concurrence. "
                   "Le mot-clé 'await' suspend une coroutine sans bloquer le thread. "
                   "asyncio.gather() permet d'exécuter plusieurs coroutines en parallèle.",
    },
    {
        "id":     "doc_2",
        "titre":  "Agents IA et boucle ReAct",
        "contenu": "Un agent ReAct alterne entre raisonnement (Thought) et action (Action). "
                   "Il appelle des outils pour obtenir des informations (Observation). "
                   "La boucle continue jusqu'à avoir une réponse finale (Final Answer). "
                   "ReAct signifie Reasoning and Acting en anglais.",
    },
    {
        "id":     "doc_3",
        "titre":  "Tool use dans les LLMs",
        "contenu": "Le tool use (function calling) permet au LLM de demander l'exécution d'une fonction. "
                   "Le développeur fournit un schéma JSON décrivant les outils disponibles. "
                   "Le LLM répond avec un tool_call contenant le nom et les arguments. "
                   "L'agent exécute la fonction et renvoie le résultat au LLM.",
    },
    {
        "id":     "doc_4",
        "titre":  "OpenRouter et les LLMs",
        "contenu": "OpenRouter est un proxy unifié pour accéder à de nombreux LLMs. "
                   "Il supporte Claude (Anthropic), GPT-4 (OpenAI), Mistral, Llama, et plus. "
                   "L'API est compatible avec le format OpenAI, donc le SDK openai fonctionne. "
                   "Les clés API commencent par 'sk-or-'. Le prix varie selon le modèle.",
    },
    {
        "id":     "doc_5",
        "titre":  "Pydantic v2",
        "contenu": "Pydantic valide les données Python avec des annotations de type. "
                   "BaseModel est la classe de base pour définir des schémas. "
                   "Field() permet d'ajouter des contraintes (min, max, description). "
                   "model_json_schema() génère automatiquement un JSON Schema — utile pour les tools.",
    },
    {
        "id":     "doc_6",
        "titre":  "LangGraph",
        "contenu": "LangGraph modélise un agent comme un graphe d'états orienté. "
                   "Chaque nœud est une fonction qui transforme l'état. "
                   "Les arcs conditionnels permettent de router vers différents nœuds. "
                   "L'état est le seul objet partagé entre tous les nœuds du graphe.",
    },
    {
        "id":     "doc_7",
        "titre":  "FastAPI",
        "contenu": "FastAPI est un framework web Python moderne et rapide. "
                   "Il génère automatiquement la documentation Swagger depuis les type hints. "
                   "Les endpoints sont des fonctions async def décorées avec @app.get/@app.post. "
                   "Pydantic est intégré nativement pour la validation des requêtes et réponses.",
    },
]


# ─── 2. CHUNKING — découper les documents ────────────────────────────────────

def chunker(texte: str, taille: int = 200, chevauchement: int = 50) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.
    Le chevauchement évite de couper une information en deux.
    """
    mots   = texte.split()
    chunks = []
    debut  = 0
    while debut < len(mots):
        fin = min(debut + taille, len(mots))
        chunks.append(" ".join(mots[debut:fin]))
        debut += taille - chevauchement
    return chunks


# ─── 3. EMBEDDINGS AVEC CHROMADB ─────────────────────────────────────────────
# ChromaDB génère les embeddings automatiquement avec son modèle par défaut.
# Pour la production : utiliser text-embedding-3-small (OpenAI) ou bge-m3.

class BaseConnaissances:
    """Vector store ChromaDB pour les documents de formation."""

    def __init__(self, nom: str = "agents_ia_kb"):
        self.client     = chromadb.PersistentClient(path="07_production/data/chroma")
        self.collection = self.client.get_or_create_collection(
            name=nom,
            metadata={"hnsw:space": "cosine"},
        )

    def indexer(self, documents: list[dict], force: bool = False) -> int:
        """Indexe les documents. Si force=False, skip les docs déjà indexés."""
        existants = set(self.collection.get()["ids"])
        nouveaux  = [d for d in documents if d["id"] not in existants]

        if not nouveaux and not force:
            console.print(f"[dim]{len(existants)} documents déjà indexés, rien à ajouter.[/dim]")
            return 0

        ids      = []
        textes   = []
        metadatas = []

        for doc in nouveaux:
            chunks = chunker(doc["contenu"])
            for i, chunk in enumerate(chunks):
                ids.append(f"{doc['id']}_chunk_{i}")
                textes.append(chunk)
                metadatas.append({"titre": doc["titre"], "doc_id": doc["id"], "chunk": i})

        self.collection.add(ids=ids, documents=textes, metadatas=metadatas)
        console.print(f"[green]✓[/green] {len(nouveaux)} documents indexés ({len(ids)} chunks)")
        return len(nouveaux)

    def rechercher(self, query: str, n: int = 3) -> list[dict]:
        """Recherche les chunks les plus pertinents pour une query."""
        resultats = self.collection.query(
            query_texts=[query],
            n_results=min(n, self.collection.count()),
        )

        chunks = []
        for i in range(len(resultats["ids"][0])):
            chunks.append({
                "id":       resultats["ids"][0][i],
                "texte":    resultats["documents"][0][i],
                "titre":    resultats["metadatas"][0][i]["titre"],
                "distance": resultats["distances"][0][i],
            })

        return sorted(chunks, key=lambda x: x["distance"])

    def stats(self) -> dict:
        return {"nb_chunks": self.collection.count()}


# ─── 4. AGENT RAG ─────────────────────────────────────────────────────────────

class AgentRAG:
    """Agent qui répond en s'appuyant sur la base de connaissances."""

    def __init__(self, kb: BaseConnaissances):
        self.kb     = kb
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def construire_contexte(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lignes = ["## Documents de référence :\n"]
        for c in chunks:
            lignes.append(f"### {c['titre']}")
            lignes.append(c["texte"])
            lignes.append("")
        return "\n".join(lignes)

    async def repondre(self, question: str, n_chunks: int = 3, verbose: bool = True) -> str:
        # 1. Retrieval
        chunks  = self.kb.rechercher(question, n=n_chunks)

        if verbose and chunks:
            console.print(f"[dim]Chunks trouvés : {[c['titre'] for c in chunks]}[/dim]")

        # 2. Augmented prompt
        contexte = self.construire_contexte(chunks)
        system   = f"""Tu es un assistant expert en Python et agents IA.
Réponds UNIQUEMENT en te basant sur les documents fournis.
Si l'information n'est pas dans les documents, dis-le clairement.

{contexte}"""

        # 3. Generation
        response = await self.client.chat.completions.create(
            model="anthropic/claude-haiku-4-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content or ""


# ─── DÉMONSTRATION ───────────────────────────────────────────────────────────

async def main():
    import asyncio
    console.rule("[bold]Leçon 3 — RAG[/bold]")

    # 1. Indexer les documents
    console.print("\n[bold cyan]1. Indexation[/bold cyan]")
    kb = BaseConnaissances()
    kb.indexer(DOCUMENTS)
    console.print(f"Stats : {kb.stats()}")

    # 2. Tester le retrieval
    console.print("\n[bold cyan]2. Retrieval (sans LLM)[/bold cyan]")
    queries = ["comment fonctionne asyncio ?", "qu'est-ce que ReAct ?"]
    for q in queries:
        chunks = kb.rechercher(q, n=2)
        console.print(f"\n[yellow]Query :[/yellow] {q}")
        for c in chunks:
            console.print(f"  [dim][{c['distance']:.3f}][/dim] {c['titre']} — {c['texte'][:60]}...")

    # 3. Agent RAG complet
    console.print("\n[bold cyan]3. Agent RAG (avec LLM)[/bold cyan]")
    agent = AgentRAG(kb)

    questions = [
        "Comment utiliser asyncio.gather() ?",
        "Qu'est-ce que le tool use dans les LLMs ?",
        "Comment OpenRouter est-il lié à OpenAI ?",
        "Quelle est la capitale de la France ?",   # hors base de connaissances
    ]

    for q in questions:
        console.print(f"\n[bold blue]Q :[/bold blue] {q}")
        reponse = await agent.repondre(q)
        console.print(Panel(reponse[:300], title="[green]Réponse RAG[/green]"))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
