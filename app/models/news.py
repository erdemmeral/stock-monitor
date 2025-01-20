from datetime import datetime
from typing import Optional

class NewsArticle:
    def __init__(
        self,
        title: str,
        content: str,
        source: str,
        published_at: datetime,
        url: str,
        related_symbols: list[str],
        sentiment_score: Optional[float] = None
    ):
        self.title = title
        self.content = content
        self.source = source
        self.published_at = published_at
        self.url = url
        self.related_symbols = related_symbols
        self.sentiment_score = sentiment_score

    def __str__(self):
        return f"{self.title} ({self.source}) - {self.published_at}" 