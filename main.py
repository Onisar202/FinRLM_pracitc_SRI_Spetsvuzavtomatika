import json
from src.finrlm.index.chunking import chunk_corpus
from src.finrlm.index.embed import embed_chunks
from src.finrlm.index.store import create_collection, upsert_chunks
from src.finrlm.agent.graph import run_agent


def main():

    answer = run_agent("Газпром дивиденды 2026")
    print(answer)

    
    
def index():
    
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
    main()
