from pycoingecko import CoinGeckoAPI

def get_top_coins(limit=5):
    cg = CoinGeckoAPI()
    try:
        coins = cg.get_coins_markets(vs_currency='usd', order='volume_desc', per_page=limit, page=1)
        return [{"id": c["id"], "symbol": c["symbol"], "price": c["current_price"]} for c in coins]
    except Exception as e:
        print(f"Error fetching CoinGecko data: {e}")
        return []