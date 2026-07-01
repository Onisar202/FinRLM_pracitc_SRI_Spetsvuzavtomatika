from openai import OpenAI
from src.finrlm.search import Doc
from src.finrlm.config import VLLM_BASE_URL, LLM_MODEL

_client: OpenAI | None = None

def _get_llm() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
    return _client

def _build_context(docs: list[Doc]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.source} ({doc.published_at[:10]})\n{doc.title}\n{doc.text}")
    return "\n\n---\n\n".join(parts)

def generate(query: str, docs: list[Doc]) -> str:
    context = _build_context(docs)
    prompt = f"""Ты финансовый аналитик. Ответь на вопрос пользователя, опираясь только на предоставленные документы.
После каждого утверждения указывай номер источника в квадратных скобках [N].
Если ответа в документах нет — честно скажи об этом.

Документы:
{context}

Вопрос: {query}

Ответ:"""

    response = _get_llm().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content
