from FlagEmbedding import FlagReranker
from src.finrlm.search import Doc

_reranker: FlagReranker | None = None

def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False, device="cpu")
    return _reranker


def rerank(query: str, docs: list[Doc], top_k: int=5) -> list[Doc]:
    # if not docs:
    #     return []
    
    # reranker = _get_reranker()
    # pairs = [[query, doc.text] for doc in docs]
    # scores = reranker.compute_score(pairs, normalize=True)

    # if isinstance(scores, float):
    #     scores = [scores]

    # ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    # return [doc for doc, _ in ranked[:top_k]]
    return docs[:top_k]