from rlm.core.rlm import RLM
from src.finrlm.search import search as _search_fn


def _search(query: str, company: str | None = None) -> list[dict]:
    """Поиск финансовых новостей. Возвращает список документов с title, text, source, url."""
    docs = _search_fn(query, company=company, limit=5)
    return [
        {
            "id": d.id,
            "title": d.title,
            "text": d.text[:300],
            "source": d.url,
            "published_at": d.published_at[:10]
        }
        for d in docs
    ]

def run_rlm(query: str) -> str:
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "finrlm-llm",
            "base_url": "http://localhost:8000/v1",
            "api_key": "dummy"
        },
        environment="local",
        custom_tools={"search": _search},
        max_depth=1,
        max_iterations=10,
        max_timeout=120,
        verbose=True
    )
    result = rlm.completion(query)
    return result.response