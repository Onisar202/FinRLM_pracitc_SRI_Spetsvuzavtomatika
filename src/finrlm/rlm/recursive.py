from rlm.core.rlm import RLM
from src.finrlm.search import search as _search_fn
from src.finrlm.rag.generate import generate as _generate_fn


def _rag_search(query: str, company: str | None = None) -> str:
    """RAG-поиск: находит релевантные новости и генерирует ответ по ним.
    Используй для каждого подвопроса отдельно."""
    docs = _search_fn(query, company=company, limit=5)
    if not docs:
        return "Документы по данному запросу не найдены."
    return _generate_fn(query, docs)


def run_rlm(query: str) -> str:
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "finrlm-llm",
            "base_url": "http://localhost:8000/v1",
            "api_key": "dummy"
        },
        environment="local",
        custom_tools={"rag_search": _rag_search},
        max_depth=1,
        max_iterations=10,
        max_timeout=120,
        verbose=True
    )
    result = rlm.completion(query)
    return result.response