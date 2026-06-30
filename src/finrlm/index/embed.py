from FlagEmbedding import BGEM3FlagModel

_model: BGEM3FlagModel | None = None

def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")
    return _model

def embed_chunks(chunks: list[dict], batch_size: int = 16) -> list[dict]:
    texts = [c["text"] for c in chunks]
    model = get_model()

    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True
    )

    dense = output["dense_vecs"]
    sparse = output["lexical_weights"]

    for chunk, d, s in zip(chunks, dense, sparse):
        chunk["dense"] = d.tolist()
        chunk["sparse"] = s

    return chunks