import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.finrlm.search import Doc

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        # AutoTokenizer грузит fast-токенизатор (XLMRobertaTokenizerFast),
        # это обходит баг FlagReranker с prepare_for_model.
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def rerank(query: str, docs: list[Doc], top_k: int = 5) -> list[Doc]:
    if not docs:
        return []

    tokenizer, model = _load()
    pairs = [[query, doc.text] for doc in docs]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

    ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
