import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

class BinanceClient:
    def get_lot_size(self, symbol: str):
        try:
            info = self.client.get_symbol_info(symbol)
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    step_size = float(f['stepSize'])
                    return min_qty, step_size
            return 0.0, 1.0  # fallback
        except Exception as e:
            raise RuntimeError(f"Binance LOT_SIZE fetch error: {e}")
    def log_trade(self, action: str, symbol: str, qty: float, price: float, fee: float, status: str, details: str = ""):
        import csv, os, datetime
        log_file = os.path.join(os.path.dirname(__file__), '../../trade_log.csv')
        log_file = os.path.abspath(log_file)
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp", "action", "symbol", "qty", "price", "fee", "status", "details"])
            writer.writerow([
                datetime.datetime.utcnow().isoformat(),
                action,
                symbol,
                qty,
                price,
                fee,
                status,
                details
            ])
    def get_trade_fee(self, symbol: str) -> float:
        try:
            fees = self.client.get_trade_fee(symbol=symbol)
            return float(fees[0]['takerCommission']) / 1000  # Binance returns per-mille
        except Exception as e:
            raise RuntimeError(f"Binance fee fetch error: {e}")

    def get_ticker(self, symbol: str):
        try:
            return self.client.get_ticker(symbol=symbol)
        except Exception as e:
            raise RuntimeError(f"Binance ticker error: {e}")
    def get_balance(self, asset: str = "USDT") -> float:
        try:
            balance_info = self.client.get_asset_balance(asset=asset)
            return float(balance_info['free'])
        except Exception as e:
            raise RuntimeError(f"Binance balance fetch error: {e}")

    def create_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET"):
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type=order_type,
                quantity=quantity
            )
            return order
        except Exception as e:
            raise RuntimeError(f"Binance order error: {e}")
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)

    def get_price(self, symbol: str):
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            raise RuntimeError(f"Binance price fetch error: {e}")

    # Add more methods as needed for trading, order management, etc.
