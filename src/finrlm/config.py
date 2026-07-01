import os

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "finrlm-llm")


RSS_FEEDS = [
    {"url": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",                "source": "РБК", "lang": "ru"},
    {"url": "https://www.kommersant.ru/RSS/news.xml",                           "source": "Коммерсантъ", "lang": "ru"},
    {"url": "https://www.interfax.ru/rss.asp",                                  "source": "Интерфакс", "lang": "ru"},
    {"url": "https://www.finam.ru/analysis/newsitem/rss/",                      "source": "Финам", "lang": "ru"},

    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",            "source": "CNBC", "lang": "en"},
    {"url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines","source": "MarketWatch", "lang": "en"},
    {"url": "https://finance.yahoo.com/news/rssindex",                          "source": "Yahoo Finance","lang": "en"}
]
