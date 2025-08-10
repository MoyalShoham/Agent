from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Trade(BaseModel):
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: datetime
    status: str  # 'executed', 'pending', 'failed'
    order_type: str  # 'market', 'limit', etc.
    is_paper: bool = False
    fee: Optional[float] = None
    profit: Optional[float] = None

class MarketData(BaseModel):
    symbol: str
    price: float
    volume: float
    market_cap: Optional[float]
    timestamp: datetime
    indicators: Optional[dict] = None

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    type: str
    content: dict
    timestamp: datetime
