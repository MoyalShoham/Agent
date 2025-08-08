from binance.client import Client
import os

def run_trading_agent(analysis_result):
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    client = Client(api_key, api_secret)
    coin = analysis_result["coin"]
    decision = analysis_result["decision"]
    if decision == "BUY":
        print(f"Would place BUY order for {coin} (not actually placing order in this demo!)")
        # client.order_market_buy(symbol=f"{coin.upper()}USDT", quantity=0.01)
    elif decision == "SELL":
        print(f"Would place SELL order for {coin}")
        # client.order_market_sell(symbol=f"{coin.upper()}USDT", quantity=0.01)
    else:
        print(f"HOLDING {coin}")
    return decision