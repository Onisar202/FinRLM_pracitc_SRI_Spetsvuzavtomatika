import hashlib
import trafilatura
from langdetect import detect, LangDetectException

MIN_TEXT_LEN = 200
_seen_urls: set[str] = set()


def fetch_fulltext(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded, include_comments=False, include_tables=False)


def is_duplicate(url: str) -> bool:
    key = hashlib.md5(url.encode()).hexdigest()
    if key in _seen_urls:
        return True
    _seen_urls.add(key)
    return False


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def clean_doc(doc: dict) -> dict | None:
    if is_duplicate(doc["url"]):
        return None

    fulltext = fetch_fulltext(doc["url"])
    text = fulltext or doc.get("text") or ""

    if len(text) < MIN_TEXT_LEN:
        return None

    lang = detect_lang(text)
    if lang not in ("ru", "en"):
        return None

    return {**doc, "text": text, "lang": lang}
