from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Modèles disponibles sur OpenRouter
MODELS = {
    "fast":    "anthropic/claude-haiku-4-5",
    "default": "anthropic/claude-sonnet-4-5",
    "smart":   "anthropic/claude-opus-4-5",
    # Alternatives gratuites pour tester
    "free":    "mistralai/mistral-7b-instruct:free",
}

def get_client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
