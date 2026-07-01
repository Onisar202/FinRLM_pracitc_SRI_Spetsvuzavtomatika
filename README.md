# FinRLM

Агентная Q&A система по финансовым новостям (ru/en). Гибрид **RAG** (поиск по новостям)
и **RLM** (рекурсивные вызовы инструмента для многошаговых вопросов).

## Как это работает

```
RSS → clean → chunk → BGE-M3 (dense + sparse) → Qdrant (hybrid RRF)
                                                       │
                        RLM ──(tool: rag_search)──►  search + LLM-ответ
```

- **Ingest** — RSS-фиды (РБК, Коммерсантъ, Интерфакс, Финам, CNBC, MarketWatch,
  Yahoo Finance), извлечение полного текста (trafilatura), определение языка.
- **Index** — чанки 512/64, эмбеддинги BGE-M3 (dense 1024 + sparse), Qdrant named vectors.
- **Search** — гибридный поиск dense + sparse с RRF-фьюжном.
- **RAG** — `rag_search(query)`: поиск top-5 → генерация ответа с цитатами `[N]`.
- **RLM** — рекурсивный цикл (`rlms`): вызывает `rag_search` для каждого подвопроса
  и синтезирует итоговый ответ.

## Требования

- Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)
- Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- vLLM (OpenAI-совместимый API) на `http://localhost:8000/v1`, модель `finrlm-llm`

## Запуск

```bash
uv sync
uv run main.py ingest                        # RSS → data/raw/news.jsonl
uv run main.py index                         # эмбеддинги → Qdrant
uv run main.py run "Газпром дивиденды 2026"   # RLM + RAG
```

Примеры вопросов:

```bash
uv run main.py run "Сравни дивиденды Газпрома и Норникеля"
uv run main.py run "Почему Норникель не платит дивиденды"
```
