"""Abstract exchange gateway interface for spot/futures trading."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class ExchangeGateway(ABC):
    @abstractmethod
    def get_balance(self) -> Dict[str, float]: ...

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]: ...

    @abstractmethod
    def create_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET', price: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]: ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def fetch_funding_rate(self, symbol: str) -> float: ...

    @abstractmethod
    def fetch_open_interest(self, symbol: str) -> float: ...

    @abstractmethod
    def fetch_orderbook(self, symbol: str, limit: int = 50) -> Dict[str, Any]: ...
