# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer les dépendances dans /app/.venv
RUN uv sync --frozen --no-dev


# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copier le venv et le code
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Le venv est prioritaire dans le PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Port exposé
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Démarrage
CMD ["uvicorn", "07_production.01_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
