from langchain_core.tools import tool 
from src.finrlm.search import search, Doc
from src.finrlm.rerank import rerank

def _format_docs(docs: list[Doc]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.source} ({doc.published_at[:10]})\n{doc.title}\n{doc.text}")
    return "\n\n---\n\n".join(parts)


@tool
def search_news(query: str) -> str:
    """Поиск финансовых новостей по запросу. Возвращает релевантные статьи с источниками."""
    docs = search(query, limit=20)
    top = rerank(query, docs, top_k=5)
    if not top:
        return "Документы не найдены."
    return _format_docs(top)

@tool
def search_news_by_company(query: str, company: str) -> str:
    """Поиск новостей по запросу с фильтром по компании (тикер или название)."""
    docs = search(query, company=company, limit=20)
    top = rerank(query, docs, top_k=5)
    if not top:
        return f"Документы по компании '{compile}' не найдены."
    return _format_docs(top)

TOOLS = [search_news, search_news_by_company]