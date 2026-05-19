# Ressources — Python & Agents IA

## Documentation officielle
- **Python** : https://docs.python.org/3/
- **Anthropic Claude API** : https://docs.anthropic.com/
- **OpenAI API** : https://platform.openai.com/docs/
- **Pydantic v2** : https://docs.pydantic.dev/
- **FastAPI** : https://fastapi.tiangolo.com/
- **LangChain** : https://python.langchain.com/
- **CrewAI** : https://docs.crewai.com/
- **MCP (Model Context Protocol)** : https://modelcontextprotocol.io/

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

### Appel Claude avec tool use
```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)
```

### Agent loop minimal
```python
import asyncio
import anthropic

async def agent_loop(user_input: str) -> str:
    client = anthropic.AsyncAnthropic()
    messages = [{"role": "user", "content": user_input}]
    
    while True:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        # Traiter les tool calls
        tool_results = await execute_tools(response.content)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```
