"""Binance Futures Gateway (USD-M) minimal implementation.
Reads credentials from environment variables:
 BINANCE_FUTURES_API_KEY
 BINANCE_FUTURES_API_SECRET
"""
from __future__ import annotations
import os, time, hmac, hashlib, requests, json, logging
from typing import Any, Dict, Optional, List
from trading_bot.utils.exchange_gateway import ExchangeGateway

class BinanceAPIError(Exception):
    def __init__(self, status_code: int, error_code: Optional[int], msg: str, body: str | None = None):
        self.status_code = status_code
        self.error_code = error_code
        self.msg = msg
        self.body = body
        super().__init__(f"BinanceAPIError status={status_code} code={error_code} msg={msg}")

class BinanceFuturesGateway(ExchangeGateway):
    ERROR_HELP = {
        -1021: 'Timestamp for this request is outside of the recvWindow. Local clock out of sync.',
        -2019: 'Margin is insufficient. Reduce position size or add collateral.',
        -1111: 'Precision / invalid quantity. Adjust step size or format.',
        -4164: 'Order quantity is too small. Below min notional after filters.',
        -4129: 'ReduceOnly order rejected. Position already closed or direction mismatch.',
        -2010: 'Generic trading error (often invalid parameter).',
        -4061: 'Account futures trading not enabled. Enable Futures in account settings.',
    }

    def __init__(self, recv_window: int = 5000):
        raw_key = os.getenv('BINANCE_FUTURES_API_KEY') or ''
        raw_secret = os.getenv('BINANCE_FUTURES_API_SECRET') or ''
        self.api_key = raw_key.strip().strip('"').strip("'")
        self.api_secret = raw_secret.strip().strip('"').strip("'")
        if not self.api_key or not self.api_secret:
            raise ValueError('Binance futures API keys not set in environment')
        if (self.api_key != raw_key) or (self.api_secret != raw_secret):
            logging.info("[GATEWAY] Stripped quotes/whitespace from futures API credentials.")
        logging.info(f"[GATEWAY] API key length={len(self.api_key)} secret length={len(self.api_secret)} (masked)")
        # Base URL validation / fallback
        raw_base = os.getenv('BINANCE_FUTURES_BASE_URL', 'https://fapi.binance.com').strip().strip('"').strip("'")
        invalid_pattern = any(seg in raw_base.lower() for seg in ['/en/', '/zh-cn/', '/de/', '/es/', '/futures/', 'html'])
        # Must point to API host, not marketing page. Accept domains containing fapi.binance.com, testnet.binancefuture.com
        is_api_like = ('fapi.binance.com' in raw_base.lower()) or ('testnet.binancefuture.com' in raw_base.lower())
        if invalid_pattern or not is_api_like:
            logging.warning(f"[GATEWAY] Invalid or non-API BINANCE_FUTURES_BASE_URL '{raw_base}'. Falling back to https://fapi.binance.com")
            raw_base = 'https://fapi.binance.com'
        self.base_url = raw_base.rstrip('/')
        self.recv_window = recv_window
        self._symbol_filters: Dict[str, Dict[str, Any]] = {}
        self._time_offset_ms: int = 0  # serverTime - localTime
        self._metrics: Dict[str, Any] = {
            'orders_sent': 0,
            'order_errors': 0,
            'error_codes': {},
            'latencies_ms': []
        }
        # Debug flag for signing issues
        self._debug_signing = os.getenv('FUTURES_DEBUG_SIGNING', '0') == '1'
        self._debug_signing_once = True
        self._load_exchange_info()
        # Initial time sync
        try:
            self._sync_time()
        except Exception as e:
            logging.warning(f"[GATEWAY] Initial time sync failed: {e}")

    def _load_exchange_info(self):
        try:
            data = self._get('/fapi/v1/exchangeInfo')
            for s in data.get('symbols', []):
                sym = s.get('symbol')
                if not sym:
                    continue
                filters = {}
                for f in s.get('filters', []):
                    filters[f.get('filterType')] = f
                self._symbol_filters[sym] = filters
        except Exception:
            # Non-fatal; filters will be missing
            pass

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        return self._symbol_filters.get(symbol.upper(), {})

    def get_min_notional(self, symbol: str) -> float:
        f = self.get_symbol_filters(symbol).get('MIN_NOTIONAL') or self.get_symbol_filters(symbol).get('NOTIONAL')
        if not f:
            return 0.0
        # MIN_NOTIONAL filter uses notional; field names can vary across docs
        for key in ('notional', 'minNotional'):
            if key in f:
                try:
                    return float(f[key])
                except Exception:
                    continue
        return 0.0

    def get_lot_size(self, symbol: str):
        f = self.get_symbol_filters(symbol).get('LOT_SIZE')
        if not f:
            return 0.0, 0.0, 0.0
        try:
            return float(f.get('minQty', 0)), float(f.get('stepSize', 0)), float(f.get('maxQty', 0))
        except Exception:
            return 0.0, 0.0, 0.0

    def _sync_time(self):
        r = requests.get(self.base_url + '/fapi/v1/time', timeout=5)
        r.raise_for_status()
        data = r.json()
        server_time = int(data.get('serverTime'))
        local_time = int(time.time() * 1000)
        self._time_offset_ms = server_time - local_time
        logging.info(f"[GATEWAY] Time sync offset set to {self._time_offset_ms} ms (base_url={self.base_url})")

    def _signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach timestamp/recvWindow and signature.
        IMPORTANT: Signature must be over the exact query string ORDER that is sent.
        Previously we sorted keys but sent insertion order, causing -1022. Now we keep insertion order.
        Set FUTURES_DEBUG_SIGNING=1 to log the canonical string (once) for diagnosis.
        """
        # Preserve user-provided insertion order: add timestamp & recvWindow last (still deterministic)
        params['timestamp'] = int(time.time()*1000 + self._time_offset_ms)
        params['recvWindow'] = self.recv_window
        # Build canonical string in the SAME order we will send (dict preserves insertion order in Python 3.7+)
        canonical_items = [(k, params[k]) for k in params.keys() if k != 'signature']
        qs = '&'.join(f"{k}={v}" for k, v in canonical_items)
        signature = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        if self._debug_signing and self._debug_signing_once:
            self._debug_signing_once = False
            logging.info(
                f"[GATEWAY][SIGN_DEBUG] offset_ms={self._time_offset_ms} recvWindow={self.recv_window} canonical='{qs}' sig={signature[:12]}... len={len(signature)}"
            )
        return params

    def _headers(self):
        return { 'X-MBX-APIKEY': self.api_key }

    def _handle_response(self, r: requests.Response, context: str):
        """Parse response, raising BinanceAPIError with detailed info on failure."""
        body_text = ''
        try:
            body_text = r.text
        except Exception:
            body_text = ''
        if r.status_code >= 400:
            error_code = None
            msg = f"HTTP {r.status_code}"
            try:
                data = r.json()
                error_code = data.get('code')
                msg = data.get('msg', msg)
            except Exception:
                pass
            help_msg = self.ERROR_HELP.get(error_code, 'See Binance API docs for this error code.') if error_code is not None else ''
            logging.error(f"[GATEWAY][ERROR] {context} status={r.status_code} code={error_code} msg={msg} help={help_msg} body={body_text[:300]}")
            raise BinanceAPIError(r.status_code, error_code, msg, body_text)
        try:
            return r.json()
        except Exception:
            return {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False):
        params = params or {}
        if signed:
            params = self._signed_params(params)
        try:
            r = requests.get(self.base_url + path, params=params, headers=self._headers(), timeout=10)
            return self._handle_response(r, f'GET {path}')
        except BinanceAPIError as e:
            # Retry once for timestamp drift
            if e.error_code == -1021:
                try:
                    self._sync_time()
                    params = self._signed_params(params) if signed else params
                    r2 = requests.get(self.base_url + path, params=params, headers=self._headers(), timeout=10)
                    return self._handle_response(r2, f'GET {path} retry')
                except Exception:
                    raise
            raise

    def _post(self, path: str, params: Dict[str, Any], signed: bool = True):
        if signed:
            params = self._signed_params(params)
        start = time.time()
        try:
            r = requests.post(self.base_url + path, params=params, headers=self._headers(), timeout=10)
            data = self._handle_response(r, f'POST {path}')
            self._metrics['orders_sent'] += 1
            return data
        except BinanceAPIError as e:
            self._metrics['order_errors'] += 1
            if e.error_code is not None:
                self._metrics['error_codes'][e.error_code] = self._metrics['error_codes'].get(e.error_code, 0) + 1
            # Timestamp drift retry
            if e.error_code == -1021:
                try:
                    self._sync_time()
                    params = self._signed_params({k:v for k,v in params.items() if k not in ('timestamp','signature')})
                    r2 = requests.post(self.base_url + path, params=params, headers=self._headers(), timeout=10)
                    data = self._handle_response(r2, f'POST {path} retry')
                    self._metrics['orders_sent'] += 1
                    return data
                except Exception:
                    raise
            raise
        finally:
            latency_ms = (time.time() - start) * 1000.0
            self._metrics['latencies_ms'].append(latency_ms)
            if len(self._metrics['latencies_ms']) > 1000:
                self._metrics['latencies_ms'] = self._metrics['latencies_ms'][-500:]

    def _delete(self, path: str, params: Dict[str, Any], signed: bool = True):
        if signed:
            params = self._signed_params(params)
        try:
            r = requests.delete(self.base_url + path, params=params, headers=self._headers(), timeout=10)
            return self._handle_response(r, f'DELETE {path}')
        except BinanceAPIError as e:
            if e.error_code == -1021:
                try:
                    self._sync_time()
                    params = self._signed_params({k:v for k,v in params.items() if k not in ('timestamp','signature')})
                    r2 = requests.delete(self.base_url + path, params=params, headers=self._headers(), timeout=10)
                    return self._handle_response(r2, f'DELETE {path} retry')
                except Exception:
                    raise
            raise

    def get_metrics(self) -> Dict[str, Any]:
        m = dict(self._metrics)
        if m['latencies_ms']:
            import statistics
            lats = m['latencies_ms']
            m['latency_ms_p50'] = statistics.median(lats)
            m['latency_ms_p95'] = sorted(lats)[int(0.95 * (len(lats)-1))]
            m['latency_ms_last'] = lats[-1]
        return m

    def get_balance(self):
        data = self._get('/fapi/v2/balance', signed=True)
        # Return dict of asset: walletBalance (more relevant than "balance")
        result = {}
        for item in data:
            asset = item.get('asset')
            bal = item.get('walletBalance') or item.get('balance')
            try:
                result[asset] = float(bal)
            except Exception:
                continue
        return result

    def get_position(self, symbol: str):
        pos = self._get('/fapi/v2/positionRisk', signed=True, params={})
        for p in pos:
            if p['symbol'] == symbol.upper():
                return p
        return {}

    def create_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET', price: Optional[float] = None, reduce_only: bool = False):
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
            'reduceOnly': 'true' if reduce_only else 'false'
        }
        if order_type.upper() == 'LIMIT' and price is not None:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        return self._post('/fapi/v1/order', params)

    def cancel_order(self, symbol: str, order_id: str):
        return self._delete('/fapi/v1/order', {'symbol': symbol.upper(), 'orderId': order_id})

    def fetch_open_orders(self, symbol: Optional[str] = None):
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._get('/fapi/v1/openOrders', params=params, signed=True)

    def fetch_funding_rate(self, symbol: str) -> float:
        data = self._get('/fapi/v1/premiumIndex', params={'symbol': symbol.upper()})
        return float(data.get('lastFundingRate', 0.0))

    def fetch_open_interest(self, symbol: str) -> float:
        data = self._get('/futures/data/openInterestHist', params={'symbol': symbol.upper(), 'period': '5m', 'limit': 1})
        if isinstance(data, list) and data:
            return float(data[-1].get('sumOpenInterest', 0.0))
        return 0.0

    def fetch_orderbook(self, symbol: str, limit: int = 50):
        return self._get('/fapi/v1/depth', params={'symbol': symbol.upper(), 'limit': limit})

    def format_quantity(self, symbol: str, quantity: float) -> float:
        min_qty, step_size, max_qty = self.get_lot_size(symbol)
        if step_size and step_size > 0:
            import math
            precision = 0
            s = f"{step_size}".rstrip('0')
            if '.' in s:
                precision = len(s.split('.')[-1])
            q = math.floor(quantity / step_size) * step_size
            return float(f"{q:.{precision}f}")
        return float(quantity)
