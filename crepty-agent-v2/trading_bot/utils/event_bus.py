"""Simple synchronous event bus."""
from __future__ import annotations
from typing import Callable, Dict, List, Any
import threading

_handlers: Dict[str, List[Callable[[Any], None]]] = {}
_lock = threading.Lock()


def subscribe(event_type: str, handler: Callable[[Any], None]):
    with _lock:
        _handlers.setdefault(event_type, []).append(handler)


def publish(event_type: str, payload: Any):
    with _lock:
        handlers = list(_handlers.get(event_type, []))
    for h in handlers:
        try:
            h(payload)
        except Exception:
            import logging
            logging.getLogger().exception(f"event handler error for {event_type}")
