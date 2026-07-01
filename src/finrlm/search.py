from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import models as qmodels
from src.finrlm.index.store import get_client, COLLECTION

_model: BGEM3FlagModel | None = None

def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")
    return _model


@dataclass
class Doc:
    id: str
    url: str
    source: str
    lang: str
    published_at: str
    title: str
    text: str
    score: float


def search(
    query: str,
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
) -> list[Doc]:
    model = _get_model()
    out = model.encode([query], return_dense=True, return_sparse=True)
    dense = out["dense_vecs"][0].tolist()
    sparse = out["lexical_weights"][0]  

    indices = [int(k) for k in sparse.keys()]
    values  = [float(v) for v in sparse.values()]

    must = []
    if company:
        must.append(qmodels.FieldCondition(
            key="companies",
            match=qmodels.MatchAny(any=[company]),
        ))
    if date_from:
        must.append(qmodels.FieldCondition(
            key="published_at",
            range=qmodels.DatetimeRange(gte=date_from),
        ))
    if date_to:
        must.append(qmodels.FieldCondition(
            key="published_at",
            range=qmodels.DatetimeRange(lte=date_to),
        ))

    q_filter = qmodels.Filter(must=must) if must else None

    hits = get_client().query_points(
        collection_name=COLLECTION,
        prefetch=[
            qmodels.Prefetch(query=dense, using="dense", limit=50),
            qmodels.Prefetch(
                query=qmodels.SparseVector(indices=indices, values=values),
                using="sparse", limit=50,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        query_filter=q_filter,
        limit=limit,
        with_payload=True,
    )

    return [
        Doc(
            id=h.payload["id"],
            url=h.payload["url"],
            source=h.payload["source"],
            lang=h.payload["lang"],
            published_at=h.payload["published_at"],
            title=h.payload["title"],
            text=h.payload["text"],
            score=h.score,
        )
        for h in hits.points
    ]