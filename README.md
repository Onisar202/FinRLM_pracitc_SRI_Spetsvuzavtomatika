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
uv run main.py run "Газпром дивиденды 2026"   # RLM + RAG (CLI)
```

### Веб-интерфейс

```bash
uv run streamlit run app/ui.py
```

Поле ввода → ответ с цитатами `[N]`; разворачивающийся блок «reasoning RLM»
показывает пошаговый ход рассуждения (какие `rag_search` вызывались и что нашлось).

## Docker (всё в одном)

`docker-compose.yml` поднимает три сервиса: **qdrant**, **vllm** (GPU) и **app** (Streamlit).
Нужен NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose up -d --build                        # qdrant + vllm + app
docker compose run --rm app python main.py ingest   # RSS → data/raw/news.jsonl
docker compose run --rm app python main.py index    # эмбеддинги → Qdrant
```

UI: <http://localhost:8501>

> vLLM грузит модель несколько минут — до готовности запросы вернут ошибку (UI покажет её аккуратно).
> Модель/квантизацию в сервисе `vllm` сверь со своим `start_vllm.sh`
> (по умолчанию `Qwen/Qwen3-8B-AWQ`, `awq_marlin`, fp8 KV).
>
> Образ vLLM закреплён на `v0.10.2` с `VLLM_USE_V1=0`: движок V1 требует UVA/pinned-память,
> недоступную в контейнере поверх WSL2, а V0 её не требует. Флаги
> `--enforce-eager --max-num-seqs 1 --gpu-memory-utilization 0.86` подобраны под 8-ГБ карту.

Адреса задаются через env (дефолты — `localhost`, поэтому запуск без Docker работает как прежде):
`QDRANT_HOST`, `QDRANT_PORT`, `VLLM_BASE_URL`, `LLM_MODEL`.

Примеры вопросов:

```bash
uv run main.py run "Сравни дивиденды Газпрома и Норникеля"
uv run main.py run "Почему Норникель не платит дивиденды"
```
