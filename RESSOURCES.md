# Ressources — Python & Agents IA

## Documentation officielle
- **Python** : https://docs.python.org/3/
- **OpenRouter** : https://openrouter.ai/docs — proxy unifié pour tous les LLMs (utilisé dans ce projet)
- **Anthropic Claude API** : https://docs.anthropic.com/
- **OpenAI SDK** : https://platform.openai.com/docs/ — SDK utilisé pour appeler OpenRouter
- **Pydantic v2** : https://docs.pydantic.dev/
- **FastAPI** : https://fastapi.tiangolo.com/
- **LangChain** : https://python.langchain.com/
- **CrewAI** : https://docs.crewai.com/
- **MCP (Model Context Protocol)** : https://modelcontextprotocol.io/
- **ChromaDB** : https://docs.trychroma.com/

## Cours recommandés
- **Python** : "Python for Everybody" (Coursera) — bases solides
- **Async Python** : "asyncio walkthrough" sur Real Python
- **LLM Agents** : DeepLearning.AI — "Building Agents with LangChain"
- **RAG** : DeepLearning.AI — "Building Applications with Vector Databases"

## Références clés à bookmarker

### Anthropic
- Cookbook officiel : https://github.com/anthropics/anthropic-cookbook
- Tool use examples : https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
- Prompt caching guide : https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

### Architecture agents
- ReAct paper : "ReAct: Synergizing Reasoning and Acting in Language Models"
- Chain of Thought : "Chain-of-Thought Prompting Elicits Reasoning in LLMs"

## Packages à installer (par module)

```bash
# Module 3-4 (base)
uv add anthropic openai pydantic python-dotenv httpx loguru rich typer

# Module 5 (agent from scratch)
uv add chromadb sentence-transformers

# Module 6 (frameworks)
uv add langchain langchain-anthropic langgraph llama-index crewai autogen

# Module 7 (production)
uv add fastapi uvicorn pytest pytest-asyncio docker mcp
```

## Exemples de code utiles

> **Note** : ce projet utilise **OpenRouter** via le SDK `openai` (compatible).
> Clé API dans `.env` : `OPENROUTER_API_KEY=sk-or-...`

### Appel Claude via OpenRouter avec tool use
```python
import json, os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Météo d'une ville",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = await client.chat.completions.create(
    model="anthropic/claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Quel temps à Paris ?"}],
)
```

### Agent loop minimal (OpenRouter)
```python
import asyncio, json, os
from openai import AsyncOpenAI

async def agent_loop(user_input: str) -> str:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    messages = [{"role": "user", "content": user_input}]

    while True:
        response = await client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            max_tokens=4096,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        choix = response.choices[0]

        if choix.finish_reason == "stop":
            return choix.message.content

        # Traiter les tool calls
        messages.append(choix.message)
        for tc in choix.message.tool_calls:
            result = execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```
