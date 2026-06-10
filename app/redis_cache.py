"""
Redis helper for short-lived chat answer cache.
"""
import hashlib
from typing import Optional

from app.config import RAGConfig


class RedisService:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None

        if not config.redis_enabled:
            return

        try:
            import redis
        except ImportError:
            return

        try:
            self.client = redis.Redis.from_url(config.redis_url, decode_responses=True)
            self.client.ping()
        except Exception:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    @staticmethod
    def _cache_key(session_id: str, question: str) -> str:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        return f"chat_cache:{session_id}:{digest}"

    def get_cached_answer(self, session_id: str, question: str) -> Optional[str]:
        if not self.available:
            return None
        return self.client.get(self._cache_key(session_id, question))

    def set_cached_answer(self, session_id: str, question: str, answer: str) -> None:
        if not self.available:
            return
        self.client.setex(
            self._cache_key(session_id, question),
            self.config.chat_cache_ttl_seconds,
            answer,
        )
