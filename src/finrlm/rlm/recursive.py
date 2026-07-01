import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from rlm.core.rlm import RLM
from src.finrlm.search import search as _search_fn
from src.finrlm.rerank import rerank as _rerank_fn
from src.finrlm.rag.generate import generate as _generate_fn
from src.finrlm.config import VLLM_BASE_URL, LLM_MODEL


_last_sources: list[dict] = []


def _rag_search(query: str, company: str | None = None) -> str:
    """RAG-поиск по финансовым новостям: возвращает готовый связный ответ
    на подвопрос с цитатами [N]. Выводи результат целиком, не обрезай —
    это уже финальный текст. Вызывай отдельно для каждого подвопроса."""
    docs = _search_fn(query, company=company, limit=15)
    if not docs:
        return "Документы по данному запросу не найдены."
    docs = _rerank_fn(query, docs, top_k=5)
    for d in docs:
        _last_sources.append({
            "title": d.title,
            "source": d.source,
            "url": d.url,
            "date": d.published_at[:10],
        })
    return _generate_fn(query, docs)


def get_last_sources() -> list[dict]:
    """Уникальные (по url) источники из последнего вызова run_rlm."""
    seen: set[str] = set()
    uniq: list[dict] = []
    for s in _last_sources:
        if s["url"] in seen:
            continue
        seen.add(s["url"])
        uniq.append(s)
    return uniq


_warmed = False


def _warmup() -> None:
    global _warmed
    if _warmed:
        return
    docs = _search_fn("warmup", limit=1)
    if docs:
        _rerank_fn("warmup", docs, top_k=1)
    _warmed = True


def run_rlm(query: str) -> str:
    _warmup()
    _last_sources.clear()
    task = (
        "Ты финансовый аналитик. Ответь на запрос пользователя о финансовых новостях. "
        "Для получения информации ОБЯЗАТЕЛЬНО используй инструмент rag_search(query) — "
        "вызывай его для каждого подвопроса. Не отвечай, не вызвав rag_search. "
        "Запрос может быть коротким (например, название компании) — это нормально, "
        "передай его в rag_search как поисковый запрос.\n\n"
        f"Запрос пользователя: {query}"
    )
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": LLM_MODEL,
            "base_url": VLLM_BASE_URL,
            "api_key": "dummy"
        },
        environment="local",
        custom_tools={"rag_search": _rag_search},
        max_depth=1,
        max_iterations=8,
        max_timeout=300,
        verbose=True
    )
    result = rlm.completion(task)
    return result.response