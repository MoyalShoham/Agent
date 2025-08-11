import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

class BinanceClient:
    def get_total_usdt_value(self):
        """Estimate total portfolio value in USDT (sum of all assets converted to USDT)."""
        portfolio = self.get_portfolio()
        total = 0.0
        for asset, amount in portfolio.items():
            if asset == 'USDT':
                total += amount
            else:
                symbol = f"{asset}USDT"
                try:
                    price = self.get_price(symbol)
                    total += amount * price
                except Exception:
                    continue  # skip assets without USDT pair
        return total
    def bollinger_bands(self, prices, window=20, num_std=2):
        if len(prices) < window:
            return None, None, None
        import statistics
        sma = self.simple_moving_average(prices, window)
        std = statistics.stdev(prices[-window:])
        upper = sma + num_std * std
        lower = sma - num_std * std
        return upper, sma, lower

    def backtest_and_export(self, symbol, interval='1h', window=14, limit=100, filename='backtest_results.csv'):
        signals = self.backtest_sma_strategy(symbol, interval, window, limit)
        if not signals:
            return None
        import csv
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['signal', 'index', 'price'])
            for sig in signals:
                writer.writerow(sig)
        return filename

    def plot_price_with_signals(self, symbol, interval='1h', window=14, limit=100, show_indicators=True):
        import matplotlib.pyplot as plt
        prices = self.get_historical_prices(symbol, interval, limit)
        if not prices:
            print('No price data for plotting.')
            return
        x = list(range(len(prices)))
        plt.figure(figsize=(12,6))
        plt.plot(x, prices, label='Price')
        if show_indicators:
            sma = [self.simple_moving_average(prices[:i+1], window) if i+1 >= window else None for i in range(len(prices))]
            ema = [self.exponential_moving_average(prices[:i+1], window) if i+1 >= window else None for i in range(len(prices))]
            upper, _, lower = self.bollinger_bands(prices, window)
            plt.plot(x, sma, label=f'SMA-{window}')
            plt.plot(x, ema, label=f'EMA-{window}')
            if upper and lower:
                plt.axhline(upper, color='r', linestyle='--', label='Bollinger Upper')
                plt.axhline(lower, color='g', linestyle='--', label='Bollinger Lower')
        signals = self.backtest_sma_strategy(symbol, interval, window, limit)
        if signals:
            for sig, idx, price in signals:
                color = 'green' if sig == 'BUY' else 'red'
                plt.scatter(idx, price, color=color, marker='^' if sig == 'BUY' else 'v', s=100, label=sig if idx == signals[0][1] else "")
        plt.title(f'{symbol} Price & Signals')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    def exponential_moving_average(self, prices, window=14):
        if len(prices) < window:
            return None
        ema = prices[0]
        k = 2 / (window + 1)
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        return ema

    def relative_strength_index(self, prices, window=14):
        if len(prices) < window + 1:
            return None
        gains = [max(prices[i+1] - prices[i], 0) for i in range(len(prices)-1)]
        losses = [max(prices[i] - prices[i+1], 0) for i in range(len(prices)-1)]
        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def backtest_sma_strategy(self, symbol, interval='1h', window=14, limit=100):
        prices = self.get_historical_prices(symbol, interval, limit)
        if not prices or len(prices) < window + 1:
            return None
        signals = []
        for i in range(window, len(prices)):
            sma = self.simple_moving_average(prices[:i], window)
            if prices[i-1] < sma and prices[i] > sma:
                signals.append(('BUY', i, prices[i]))
            elif prices[i-1] > sma and prices[i] < sma:
                signals.append(('SELL', i, prices[i]))
        return signals

    def get_portfolio(self):
        try:
            balances = self.client.get_account()['balances']
            portfolio = {b['asset']: float(b['free']) for b in balances if float(b['free']) > 0}
            return portfolio
        except Exception as e:
            raise RuntimeError(f"Binance portfolio fetch error: {e}")
    def get_historical_prices(self, symbol: str, interval: str = '1h', limit: int = 50):
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            closes = [float(k[4]) for k in klines]
            return closes
        except Exception as e:
            raise RuntimeError(f"Binance historical price fetch error: {e}")

    def simple_moving_average(self, prices, window=14):
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window

    def calculate_trade_pnl(self, action, entry_price, exit_price, qty, fee):
        if action == 'BUY':
            return (exit_price - entry_price) * qty - fee
        elif action == 'SELL':
            return (entry_price - exit_price) * qty - fee
        return 0
    def __init__(self, api_key=None, api_secret=None, paper_trading=None):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        from trading_bot.config.settings import settings
        self.paper_trading = paper_trading if paper_trading is not None else getattr(settings, 'PAPER_TRADING', True)

    def get_min_notional(self, symbol: str):
        try:
            info = self.client.get_symbol_info(symbol)
            for f in info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    return float(f['minNotional'])
            return 0.0
        except Exception as e:
            raise RuntimeError(f"Binance MIN_NOTIONAL fetch error: {e}")

    def format_quantity(self, qty: float, step_size: float) -> str:
        # Format qty to match allowed decimals for step_size
        import decimal
        step = decimal.Decimal(str(step_size))
        precision = abs(step.as_tuple().exponent)
        fmt_qty = f"{qty:.{precision}f}"
        return fmt_qty
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
    def log_trade(self, action: str, symbol: str, qty: float, price: float, fee: float, status: str, details: str = "", strategy: str = "unknown"):
        import csv, os, datetime
        # Always use the actual strategy name if available, fallback to 'unknown'
        actual_strategy = strategy if strategy not in [None, '', 'unknown'] else 'unknown'
        # Log to main trade log
        log_file = os.path.join(os.path.dirname(__file__), '../../trade_log.csv')
        log_file = os.path.abspath(log_file)
        file_exists = os.path.isfile(log_file)
        current_total_usdt = None
        try:
            current_total_usdt = round(self.get_total_usdt_value(), 4)
        except Exception:
            current_total_usdt = ''
        def fmt(val):
            if isinstance(val, float):
                return f"{val:.5e}"
            return val
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp", "action", "symbol", "qty", "price", "fee", "status", "details", "strategy", "current_total_usdt"])
            writer.writerow([
                datetime.datetime.utcnow().isoformat(),
                action,
                symbol,
                fmt(qty),
                fmt(price),
                fmt(fee),
                status,
                details,
                actual_strategy,
                fmt(current_total_usdt)
            ])
        # Also log to the strategy-specific log file for ML/analytics
        strat_log_file = os.path.join(os.path.dirname(__file__), '../../trade_log_clean_fixed_with_strategy.csv')
        strat_log_file = os.path.abspath(strat_log_file)
        strat_exists = os.path.isfile(strat_log_file)
        with open(strat_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not strat_exists:
                writer.writerow(["timestamp", "action", "symbol", "qty", "price", "fee", "status", "details", "strategy"])
            writer.writerow([
                datetime.datetime.utcnow().isoformat(),
                action,
                symbol,
                qty,
                price,
                fee,
                status,
                details,
                actual_strategy
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
        # Adjust quantity to valid LOT_SIZE and MIN_NOTIONAL before placing order
        min_qty, step_size = self.get_lot_size(symbol)
        min_notional = self.get_min_notional(symbol)
        import math, logging
        price = self.get_price(symbol)
        # Calculate minimum qty for both minQty and minNotional
        precision = int(round(-math.log10(step_size))) if step_size < 1 else 0
        min_qty_for_notional = math.ceil(min_notional / price / step_size) * step_size
        min_qty_for_notional = round(min_qty_for_notional, precision)
        # Use the greater of min_qty and min_qty_for_notional
        required_qty = max(min_qty, min_qty_for_notional)
        # If requested quantity is too low, bump to required_qty
        adj_qty = max(quantity, required_qty)
        # Round down to step size
        adj_qty = math.floor(adj_qty / step_size) * step_size
        adj_qty = round(adj_qty, precision)
        notional = adj_qty * price
        # Final checks
        if adj_qty < min_qty or notional < min_notional or adj_qty == 0:
            logging.warning(f"[Order] {symbol}: qty {adj_qty} < min_qty {min_qty} or notional {notional} < min_notional {min_notional}. Skipping order.")
            raise RuntimeError(f"Order does not meet Binance minQty/minNotional for {symbol}: qty={adj_qty}, notional={notional}")
        if self.paper_trading:
            # Simulate order
            import random, datetime
            fake_order = {
                'symbol': symbol,
                'orderId': random.randint(10000000, 99999999),
                'side': side.upper(),
                'type': order_type,
                'status': 'FILLED',
                'price': str(price),
                'origQty': str(adj_qty),
                'executedQty': str(adj_qty),
                'transactTime': int(datetime.datetime.utcnow().timestamp() * 1000),
                'paper': True
            }
            logging.info(f"[Paper Order] {symbol}: qty={adj_qty}, price={price}, notional={notional}")
            return fake_order
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type=order_type,
                quantity=adj_qty
            )
            logging.info(f"[Real Order] {symbol}: qty={adj_qty}, price={price}, notional={notional}")
            return order
        except Exception as e:
            logging.error(f"[Order Error] {symbol}: {e}")
            raise RuntimeError(f"Binance order error: {e}")
    # (Removed duplicate __init__)

    def get_price(self, symbol: str):
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            raise RuntimeError(f"Binance price fetch error: {e}")

    # Add more methods as needed for trading, order management, etc.
