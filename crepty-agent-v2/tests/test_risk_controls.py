
from trading_bot.utils.risk_controls import check_circuit_breaker

def test_check_circuit_breaker_triggers_absolute():
    assert check_circuit_breaker(-1000, -500) is True

def test_check_circuit_breaker_not_triggers_absolute():
    assert check_circuit_breaker(-100, -500) is False

def test_check_circuit_breaker_triggers_fractional():
    assert check_circuit_breaker(-250, 0.2, starting_balance=1000) is True

def test_check_circuit_breaker_not_triggers_fractional():
    assert check_circuit_breaker(-100, 0.2, starting_balance=1000) is False

