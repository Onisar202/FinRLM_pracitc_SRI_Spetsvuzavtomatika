def chunk_doc(doc: dict, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    text = doc["text"]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "id": f"{doc['id']}_c{len(chunks)}",
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
        start += chunk_size - overlap

    return chunks


def chunk_corpus(docs: list[dict]) -> list[dict]:
    result = []
    for doc in docs:
        result.extend(chunk_doc(doc))
    return result
