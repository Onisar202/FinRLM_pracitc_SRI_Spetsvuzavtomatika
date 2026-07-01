import sys
import json

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from src.finrlm.rlm.recursive import run_rlm, get_last_sources


def run(query: str) -> None:
    answer = run_rlm(query)
    print(answer)

    sources = get_last_sources()
    if sources:
        print("\nИсточники:")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['source']} ({s['date']}): {s['url']}")


def ingest() -> None:
    from src.finrlm.config import RSS_FEEDS
    from src.finrlm.ingest.rss import fetch_feed
    from src.finrlm.ingest.clean import clean_doc
    from src.finrlm.index.store import save_docs

    all_docs = []
    for feed in RSS_FEEDS:
        for doc in fetch_feed(feed):
            cleaned = clean_doc(doc)
            if cleaned is not None:
                all_docs.append(cleaned)
    print(f"Собрано: {len(all_docs)} документов")

    save_docs(all_docs, "data/raw/news.jsonl")
    print("Сохранено в data/raw/news.jsonl")


def index() -> None:
    from src.finrlm.index.chunking import chunk_corpus
    from src.finrlm.index.embed import embed_chunks
    from src.finrlm.index.store import create_collection, upsert_chunks

    docs = []
    with open("data/raw/news.jsonl", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"Загружено: {len(docs)} документов")

    chunks = chunk_corpus(docs)
    print(f"Чанков: {len(chunks)}")

    chunks = embed_chunks(chunks)
    print("Эмбеддинги готовы")

    create_collection()
    upsert_chunks(chunks)
    print("Залито в Qdrant")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Газпром дивиденды 2026"

    if mode == "ingest":
        ingest()
    elif mode == "index":
        index()
    else:
        run(query)
