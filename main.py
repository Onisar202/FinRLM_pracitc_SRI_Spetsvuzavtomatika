import json
from src.finrlm.index.chunking import chunk_corpus
from src.finrlm.index.embed import embed_chunks
from src.finrlm.index.store import create_collection, upsert_chunks
from src.finrlm.search import search
from src.finrlm.rerank import rerank
from src.finrlm.rag.generate import generate1


def main():
    docs = search("Газпром дивиденды 2026", limit=20)
    top = rerank("Газпром дивиденды 2026", docs, top_k=5)

    

    answer = generate1("Газпром дивиденды 2026", top)
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
