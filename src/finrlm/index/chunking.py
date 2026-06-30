

def chunk_doc(doc: dict, chink_size: int = 512, overlap: int = 64) -> list[dict]:
    text = doc["text"]
    chinks = []
    start = 0
    while start < len(text):
        end = start + chink_size
        chunk_text = text[start:end]
        chinks.append({
            "id": f"{doc['id']}_c{len(chinks)}",
            "doc_id": doc["id"],
            "url": doc["url"],
            "source": doc["source"],
            "lang": doc["lang"],
            "published_at": doc["published_at"],
            "title": doc["title"],
            "text": chunk_text,
            "companies": doc["companies"],
            "tickers": doc["tickers"],
        })
        start += chink_size - overlap

    return chinks

def chunk_corpus(docs: list[dict]) -> list[dict]:
    result = []
    for doc in docs:
        result.extend(chunk_doc(doc))
    return result