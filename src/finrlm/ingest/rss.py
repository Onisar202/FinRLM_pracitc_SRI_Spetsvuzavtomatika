import feedparser
import hashlib
from datetime import datetime, timezone

def fetch_feed(feed: dict) -> list[dict]:
    parsed = feedparser.parse(feed["url"])
    results = []

    for enrty in parsed.entries:
        url = enrty.get("link", "")
        title = enrty.get("title", "")
        text = enrty.get("summary")
        pub = enrty.get("published_parsed")
        published_at = (
            datetime(*pub[:6], tzinfo=timezone.utc).isoformat()
            if pub else datetime.now(timezone.utc).isoformat()
        )
        uid = hashlib.md5(url.encode()).hexdigest()[:6]
        date_str = published_at[:10]

        results.append({
            "id": f"{feed['source'].lower()}_{date_str}_{uid}",
            "url": url,
            "source": feed["source"],
            "lang": feed["lang"],
            "published_at": published_at,
            "title": title,
            "text": text,
            "companies": [],
            "tickers": []
        })

    return results
