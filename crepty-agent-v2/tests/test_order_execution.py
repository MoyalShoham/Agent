import pytest
from trading_bot.utils.order_execution import twap_order

def test_twap_order_basic():
    symbol = "BTCUSDT"
    qty = 1.0
    price = 30000.0
    orders = twap_order(symbol, qty, price)
    assert isinstance(orders, list)
    assert all(len(order) == 3 for order in orders)
    assert all(order[0] == symbol for order in orders)

def test_twap_order_zero_qty():
    symbol = "BTCUSDT"
    qty = 0.0
    price = 30000.0
    orders = twap_order(symbol, qty, price)
    assert orders == []
