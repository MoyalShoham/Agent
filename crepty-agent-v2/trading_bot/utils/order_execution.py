"""
Order Execution Module - unified spot/futures with position sizing.
"""
from __future__ import annotations
from typing import Optional, Dict
import logging
from trading_bot.config.settings import settings
from trading_bot.execution.position_manager import PositionManager
from trading_bot.utils.event_bus import publish
from http import HTTPStatus
from math import floor
import os

try:
    from trading_bot.exchange.binance_futures_gateway import BinanceFuturesGateway, BinanceAPIError
except Exception:
    try:
        from trading_bot.exchange.binance_futures_gateway import BinanceFuturesGateway
        BinanceAPIError = Exception  # fallback
    except Exception:
        BinanceFuturesGateway = None
        BinanceAPIError = Exception

_position_manager: PositionManager | None = None
_gateway = None
_initialized = False
_symbol_min_notional: dict[str, float] = {}


def initialize(external_equity: float | None = None):
    global _position_manager, _gateway, _initialized, _symbol_min_notional
    if _initialized:
        return
    if settings.FUTURES_ENABLED and BinanceFuturesGateway:
        _gateway = BinanceFuturesGateway()
        symbols = [s.strip().upper() for s in settings.FUTURES_SYMBOLS.split(',') if s.strip()]
        _position_manager = PositionManager(symbols, max_leverage=settings.MAX_LEVERAGE, risk_per_trade=settings.RISK_PER_TRADE)
        # Fetch futures wallet equity (USDT) directly
        try:
            balances = _gateway.get_balance()
            fut_equity = float(balances.get('USDT', 0.0))
            if fut_equity > 0:
                _position_manager.update_equity(fut_equity)
                logging.info(f"[INIT] Futures wallet USDT equity set to {fut_equity}")
            elif external_equity is not None:
                _position_manager.update_equity(external_equity)
        except Exception as e:
            logging.warning(f"[INIT] Could not fetch futures balance: {e}; falling back to external equity")
            if external_equity is not None:
                _position_manager.update_equity(external_equity)
        # Cache min notional per symbol
        for sym in symbols:
            try:
                mn = _gateway.get_min_notional(sym)
                _symbol_min_notional[sym] = mn
            except Exception:
                _symbol_min_notional[sym] = 0.0
        logging.info(f"OrderExecution initialized for futures symbols={symbols}")
    else:
        logging.info("OrderExecution running in spot-only mode (legacy path).")
    _initialized = True


def update_equity(equity: float):
    if _position_manager:
        _position_manager.update_equity(equity)


def submit_signal(symbol: str, signal: str, mark_price: float, atr: float | None = None) -> Optional[Dict]:
    if not _initialized:
        initialize()
    if not settings.FUTURES_ENABLED or _gateway is None or _position_manager is None:
        logging.debug("Futures disabled or gateway not available; skipping submit_signal.")
        return None
    # --- Pyramiding config (env driven) ---
    pyramid_enabled = os.getenv('PYRAMID_ENABLED', '1') == '1'
    pyramid_layers = int(os.getenv('PYRAMID_LAYERS', '3'))
    pyramid_increment = float(os.getenv('PYRAMID_INCREMENT', '0.5'))  # +50% base per added layer
    tp_atr_mult = float(os.getenv('PYRAMID_TP_ATR_MULT', '2.0'))      # scale out trigger (unrealized / (atr))
    scale_out_one = os.getenv('PYRAMID_SCALE_OUT_MODE', 'one') == 'one'  # remove one layer at a time
    # Track invalid symbols (module-level cache)
    global _invalid_symbols
    try:
        _invalid_symbols
    except NameError:
        _invalid_symbols = {}
    if symbol in _invalid_symbols and _invalid_symbols[symbol] >= 2:
        logging.debug(f"[SKIP] {symbol} previously invalid (-1121); skipping further attempts.")
        return None
    # Skip if notional can never reach min notional with max leverage * equity
    min_notional = _symbol_min_notional.get(symbol.upper(), 0.0)
    theoretical_max_notional = _position_manager.equity * settings.MAX_LEVERAGE
    if min_notional and theoretical_max_notional < min_notional:
        logging.debug(f"[SKIP] {symbol} theoretical_max_notional {theoretical_max_notional:.4f} < min_notional {min_notional:.4f}")
        return None
    pos = _position_manager.get_position(symbol)
    base_target = 0.0
    if signal in ('buy','sell'):
        base_qty = _position_manager.target_position_size(symbol, mark_price, atr=atr)
        direction = 1 if signal == 'buy' else -1
        base_target = direction * base_qty
    # Hold -> potential scale-out evaluation (only if pyramiding & multiple layers & unrealized TP)
    if signal == 'hold' and pyramid_enabled and pos.size != 0 and atr and atr > 0:
        unreal = pos.unrealized_pnl(mark_price)
        # risk unit ~ atr * position_size_unit -> approximate using atr
        if unreal / atr >= tp_atr_mult:
            # reduce by one layer (if layered)
            base_unit = _position_manager.target_position_size(symbol, mark_price, atr=atr)
            if base_unit > 0:
                current_layers = int(abs(pos.size) / base_unit + 1e-9)
                target_layers = current_layers - 1 if scale_out_one else 1
                new_abs = base_unit * (1 + pyramid_increment * (target_layers-1)) if target_layers > 0 else 0
                desired_target = new_abs * (1 if pos.size > 0 else -1)
                if abs(desired_target - pos.size) / max(abs(pos.size), 1e-9) > 0.05:  # meaningful change
                    logging.info(f"[PYRAMID][TP] Scaling out {symbol}: layers {current_layers}->{target_layers} unreal/ATR={unreal/atr:.2f}")
                    # Forge synthetic signal to adjust toward desired_target
                    signal = 'buy' if desired_target > pos.size else 'sell'
                    base_target = desired_target
    # Pyramiding for same-direction continuation
    if pyramid_enabled and signal in ('buy','sell') and base_target != 0:
        base_unit = _position_manager.target_position_size(symbol, mark_price, atr=atr)
        if base_unit > 0:
            current_layers = int(abs(pos.size) / base_unit + 1e-9) if pos.size * base_target > 0 else 0
            # criteria: same direction continuation & unrealized >= 0 (avoid adding to losers)
            unreal = pos.unrealized_pnl(mark_price) if pos.size * base_target > 0 else 0.0
            can_add = (pos.size * base_target > 0) and (current_layers < pyramid_layers) and (unreal >= 0)
            if can_add:
                target_layers = current_layers + 1
                # compute pyramided absolute target
                pyramided_abs = base_unit * (1 + pyramid_increment * (target_layers-1))
                base_target = pyramided_abs * (1 if base_target > 0 else -1)
                logging.debug(f"[PYRAMID] {symbol} layering {current_layers}->{target_layers} base_unit={base_unit:.6f} new_target={base_target:.6f} unreal={unreal:.4f}")
    # Derive final desired target qty (qty_target)
    if signal == 'buy':
        qty_target = base_target if base_target else _position_manager.target_position_size(symbol, mark_price, atr=atr)
    elif signal == 'sell':
        qty_target = base_target if base_target else -_position_manager.target_position_size(symbol, mark_price, atr=atr)
    else:
        qty_target = 0.0
    delta = qty_target - pos.size
    if abs(delta) < 1e-9:
        return None
    side = 'BUY' if delta > 0 else 'SELL'
    order_qty = abs(delta)
    # Final min notional check with current price
    notional = order_qty * mark_price
    if min_notional and notional < min_notional:
        logging.debug(f"[SKIP] {symbol} order notional {notional:.4f} < min_notional {min_notional:.4f}")
        return None
    # --- Quantity normalization per lot size ---
    try:
        if hasattr(_gateway, 'get_lot_size'):
            min_qty, step_size, max_qty = _gateway.get_lot_size(symbol)
            if step_size and step_size > 0:
                # floor to step
                raw_qty = order_qty
                order_qty = floor(order_qty / step_size) * step_size
                # precision rounding based on step_size decimals
                step_decimals = 0
                if '.' in f"{step_size}":
                    step_decimals = len(f"{step_size}".rstrip('0').split('.')[-1])
                order_qty = float(f"{order_qty:.{step_decimals}f}")
                if raw_qty != order_qty:
                    logging.debug(f"[NORM] {symbol} qty adjusted {raw_qty} -> {order_qty} (step={step_size})")
                if min_qty and order_qty < min_qty:
                    logging.debug(f"[SKIP] {symbol} adjusted qty {order_qty} < minQty {min_qty}")
                    return None
                if max_qty and order_qty > max_qty:
                    order_qty = max_qty
            if order_qty <= 0:
                return None
    except Exception as norm_e:
        logging.debug(f"[NORM] {symbol} normalization error ignored: {norm_e}")
    # Recompute notional after normalization
    notional = order_qty * mark_price
    if min_notional and notional < min_notional:
        logging.debug(f"[SKIP] {symbol} post-normalization notional {notional:.4f} < min_notional {min_notional:.4f}")
        return None
    prev_pos_size = pos.size
    publish('ORDER_SUBMITTED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'signal': signal, 'atr': atr, 'target_qty': qty_target})
    try:
        order = _gateway.create_order(symbol, side, order_qty, order_type='MARKET', reduce_only=(side=='SELL' and qty_target==0))
        # Assume immediate fill for simplicity (enhance with status polling later)
        fill_qty = order_qty if side == 'BUY' else -order_qty
        pos.update_fill(fill_qty, mark_price)
        publish('ORDER_FILLED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'price': mark_price, 'new_pos': pos.size, 'prev_pos': prev_pos_size, 'signal': signal, 'atr': atr, 'target_qty': qty_target, 'order_id': order.get('orderId'), 'realized_pnl': pos.realized_pnl})
        publish('POSITION_CHANGED', {'symbol': symbol, 'size': pos.size, 'entry': pos.entry_price, 'realized': pos.realized_pnl})
        return order
    except BinanceAPIError as be:
        # Granular handling
        err_code = getattr(be, 'error_code', None)
        msg = getattr(be, 'msg', str(be))
        logging.warning(f"[ORDER] BinanceAPIError symbol={symbol} code={err_code} msg={msg}")
        # Auto-mark invalid symbols and prevent spam
        if err_code == -1121:  # Invalid symbol
            _invalid_symbols[symbol] = _invalid_symbols.get(symbol, 0) + 1
            if _invalid_symbols[symbol] == 1:
                logging.error(f"[SYMBOL] {symbol} marked invalid (first occurrence). Will retry once more before removal.")
            elif _invalid_symbols[symbol] >= 2:
                logging.error(f"[SYMBOL] {symbol} marked invalid twice. Removing from active trading list.")
                try:
                    # Remove from position manager symbols (optional cleanup)
                    if symbol in _position_manager.positions:
                        del _position_manager.positions[symbol]
                except Exception:
                    pass
        adjusted = False
        # Precision / quantity too small -> adjust using format_quantity
        if err_code in (-1111, -4164) or ('precision' in msg.lower()):
            try:
                fq = _gateway.format_quantity(symbol, order_qty * 0.999)  # slight shrink
                if fq and fq != order_qty:
                    logging.info(f"[ORDER][RETRY] Adjusting qty {order_qty} -> {fq} for precision/lot size")
                    order_qty = fq
                    adjusted = True
            except Exception:
                pass
        # Insufficient margin -> halve size
        if err_code == -2019:
            new_qty = order_qty * 0.5
            if new_qty * mark_price * settings.MAX_LEVERAGE >= min_notional and new_qty >= 0:
                logging.info(f"[ORDER][RETRY] Reducing qty {order_qty} -> {new_qty} due to insufficient margin")
                order_qty = new_qty
                adjusted = True
        # ReduceOnly rejection -> try without reduceOnly (only if we attempted closing)
        if err_code == -4129 and (side=='SELL' and qty_target==0) and pos.size != 0:
            logging.info("[ORDER][RETRY] Removing reduceOnly flag and retrying")
            try:
                order = _gateway.create_order(symbol, side, order_qty, order_type='MARKET', reduce_only=False)
                fill_qty = order_qty if side == 'BUY' else -order_qty
                pos.update_fill(fill_qty, mark_price)
                publish('ORDER_FILLED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'price': mark_price, 'new_pos': pos.size, 'prev_pos': prev_pos_size, 'signal': signal, 'atr': atr, 'target_qty': qty_target, 'order_id': order.get('orderId'), 'realized_pnl': pos.realized_pnl})
                publish('POSITION_CHANGED', {'symbol': symbol, 'size': pos.size, 'entry': pos.entry_price, 'realized': pos.realized_pnl})
                return order
            except Exception as e2:
                logging.warning(f"[ORDER][RETRY] reduceOnly removal failed: {e2}")
        if adjusted:
            try:
                order = _gateway.create_order(symbol, side, order_qty, order_type='MARKET', reduce_only=(side=='SELL' and qty_target==0))
                fill_qty = order_qty if side == 'BUY' else -order_qty
                pos.update_fill(fill_qty, mark_price)
                publish('ORDER_FILLED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'price': mark_price, 'new_pos': pos.size, 'prev_pos': prev_pos_size, 'signal': signal, 'atr': atr, 'target_qty': qty_target, 'order_id': order.get('orderId'), 'realized_pnl': pos.realized_pnl})
                publish('POSITION_CHANGED', {'symbol': symbol, 'size': pos.size, 'entry': pos.entry_price, 'realized': pos.realized_pnl})
                return order
            except Exception as retry_e:
                logging.warning(f"[ORDER][RETRY_FAIL] symbol={symbol} code={err_code} retry_error={retry_e}")
        # If we reach here, reject (no simulation unless paper/auth)
        if settings.PAPER_TRADING:
            sim_id = f"SIM-{symbol}-{int(__import__('time').time()*1000)}"
            fill_qty = order_qty if side == 'BUY' else -order_qty
            pos.update_fill(fill_qty, mark_price)
            publish('ORDER_FILLED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'price': mark_price, 'new_pos': pos.size, 'prev_pos': prev_pos_size, 'signal': signal, 'atr': atr, 'target_qty': qty_target, 'order_id': sim_id, 'realized_pnl': pos.realized_pnl, 'simulated': True})
            publish('POSITION_CHANGED', {'symbol': symbol, 'size': pos.size, 'entry': pos.entry_price, 'realized': pos.realized_pnl})
            return {'orderId': sim_id, 'symbol': symbol, 'status': 'FILLED', 'executedQty': order_qty, 'price': mark_price, 'simulated': True}
        publish('ORDER_REJECTED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'error': msg, 'error_code': err_code, 'signal': signal, 'atr': atr})
        return None
    except Exception as e:
        logging.exception(f"submit_signal error {symbol}: {e}")
        # Simulated fill fallback if paper trading OR auth error
        auth_fail = any(code in str(e) for code in ['401', '403'])
        if settings.PAPER_TRADING or auth_fail:
            sim_id = f"SIM-{symbol}-{int(__import__('time').time()*1000)}"
            fill_qty = order_qty if side == 'BUY' else -order_qty
            pos.update_fill(fill_qty, mark_price)
            publish('ORDER_FILLED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'price': mark_price, 'new_pos': pos.size, 'prev_pos': prev_pos_size, 'signal': signal, 'atr': atr, 'target_qty': qty_target, 'order_id': sim_id, 'realized_pnl': pos.realized_pnl, 'simulated': True})
            publish('POSITION_CHANGED', {'symbol': symbol, 'size': pos.size, 'entry': pos.entry_price, 'realized': pos.realized_pnl})
            return {'orderId': sim_id, 'symbol': symbol, 'status': 'FILLED', 'executedQty': order_qty, 'price': mark_price, 'simulated': True}
        publish('ORDER_REJECTED', {'symbol': symbol, 'side': side, 'qty': order_qty, 'error': str(e), 'signal': signal, 'atr': atr})
        return None

# Backwards compatibility TWAP helper

def twap_order(symbol, qty, price, n_slices=5):
    # Return empty if qty is zero or negative
    if qty <= 0:
        return []
    slice_qty = qty / n_slices
    return [(symbol, slice_qty, price) for _ in range(n_slices)]
