
from datetime import datetime

class NewsArticle:
    def __init__(self, title: str, content: str, published_at: datetime):
        self.title = title
        self.content = content
        self.published_at = published_at
