from binance.client import Client
import os

def run_trading_agent(analysis_result):
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ Binance API keys not found in .env file")
        return "ERROR: Missing API keys"
    
    client = Client(api_key, api_secret)
    coin = analysis_result["coin"]
    decision = analysis_result["decision"]
    
    # Get account balance
    try:
        account = client.get_account()
        usdt_balance = float([asset['free'] for asset in account['balances'] if asset['asset'] == 'USDT'][0])
        print(f"💰 USDT Balance: {usdt_balance}")
    except Exception as e:
        print(f"❌ Error getting account info: {e}")
        return "ERROR: Cannot access account"
    
    # Trading logic
    if decision == "BUY" and usdt_balance > 10:  # Only buy if we have more than $10 USDT
        try:
            symbol = f"{coin.upper()}USDT"
            # Place a small market buy order (adjust quantity as needed)
            order = client.order_market_buy(symbol=symbol, quoteOrderQty=10)  # $10 worth
            print(f"✅ BUY order placed for {symbol}: {order}")
            return f"BUY executed for {symbol}"
        except Exception as e:
            print(f"❌ Error placing BUY order: {e}")
            return f"BUY failed: {e}"
            
    elif decision == "SELL":
        try:
            symbol = f"{coin.upper()}USDT"
            # Get current coin balance
            coin_balance = float([asset['free'] for asset in account['balances'] if asset['asset'] == coin.upper()][0])
            if coin_balance > 0:
                order = client.order_market_sell(symbol=symbol, quantity=coin_balance)
                print(f"✅ SELL order placed for {symbol}: {order}")
                return f"SELL executed for {symbol}"
            else:
                print(f"⚠️ No {coin.upper()} balance to sell")
                return f"No {coin.upper()} to sell"
        except Exception as e:
            print(f"❌ Error placing SELL order: {e}")
            return f"SELL failed: {e}"
    else:
        print(f"💤 HOLDING {coin}")
        return f"HOLDING {coin}"