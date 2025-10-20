from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, Dict, Iterable, List

from app.schemas import Message


class ConversationMemory:
    """Thread-safe in-memory store that keeps the most recent chat messages."""

    def __init__(self, max_messages: int) -> None:
        self._max_messages = max_messages
        self._store: Dict[str, Deque[Message]] = {}
        self._lock = Lock()

    def get_history(self, conversation_id: str) -> List[Message]:
        with self._lock:
            if conversation_id not in self._store:
                return []
            return list(self._store[conversation_id])

    def set_history(self, conversation_id: str, messages: Iterable[Message]) -> None:
        with self._lock:
            limited = deque(maxlen=self._max_messages)
            for message in messages:
                limited.append(message)
            self._store[conversation_id] = limited

    def append_messages(
        self, conversation_id: str, messages: Iterable[Message]
    ) -> None:
        with self._lock:
            history = self._store.setdefault(
                conversation_id, deque(maxlen=self._max_messages)
            )
            for message in messages:
                history.append(message)

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._store.pop(conversation_id, None)
