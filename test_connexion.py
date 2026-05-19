"""Lance ce fichier pour vérifier que ta clé OpenRouter fonctionne."""
from config import get_client, MODELS

client = get_client()

response = client.chat.completions.create(
    model=MODELS["default"],
    messages=[{"role": "user", "content": "Dis juste 'Connexion OK !'"}],
    max_tokens=20,
)

print(response.choices[0].message.content)
