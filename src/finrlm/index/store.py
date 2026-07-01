import json, pathlib
import os

from qdrant_client import QdrantClient, models

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

COLLECTION = "finrlm"
DENSE_DIM = 1024

_client: QdrantClient | None = None


def save_docs(docs: list[dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host="localhost", port=6333)
    return _client


def create_collection() -> None:
    client = get_client()
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )


def upsert_chunks(chunks: list[dict], batch_size: int = 256) -> None:
    client = get_client()
    points = []
    for i, chunk in enumerate(chunks):
        sparse = chunk["sparse"]
        indices = [int(k) for k in sparse.keys()]
        values  = [float(v) for v in sparse.values()]
        points.append(models.PointStruct(
            id=i,
            vector={
                "dense": chunk["dense"],
                "sparse": models.SparseVector(indices=indices, values=values),
            },
            payload={k: chunk[k] for k in ("id", "doc_id", "url", "source", "lang",
                                             "published_at", "title", "text",
                                             "companies", "tickers")},
        ))

    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION,
            points=points[i : i + batch_size],
            wait=True,
        )
