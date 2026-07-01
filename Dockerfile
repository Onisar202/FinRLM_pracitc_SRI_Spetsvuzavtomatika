# FinRLM app (Streamlit + RAG/RLM). GPU не нужен — BGE-M3 и reranker считаются на CPU,
# LLM обслуживает отдельный контейнер vLLM.
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    HF_HOME=/app/.hf_cache \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# git нужен uv для установки rlms (git-зависимость).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Слой зависимостей: кэшируется, пока не менялись pyproject.toml / uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Исходники приложения (проект не устанавливается как пакет — запускается из /app).
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
