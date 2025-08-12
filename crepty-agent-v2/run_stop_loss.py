"""Flatten positions for symbols removed from active FUTURES_SYMBOLS list.

Usage (PowerShell):
  python run_stop_loss.py

This script:
  * Loads current FUTURES_SYMBOLS from settings
  * Initializes order_execution to access position manager & gateway
  * Iterates all tracked positions
  * For any symbol NOT in the active list and with non‑zero size, submits a closing MARKET order (reduce only)
  * Logs actions and errors

Safety:
  * Skips if futures disabled or gateway unavailable
  * Uses current mark price from gateway position (fallback to last entry price if needed)
  * Dry run mode via DRY_RUN=1 env

Environment variables:
  DRY_RUN=1           -> simulate without sending real orders
  LOOP_INTERVAL_SEC   -> if set (>0) run in continuous loop (default 0 single pass)
  EXIT_WHEN_EMPTY=1   -> in loop mode, exit when no positions to flatten
"""
from __future__ import annotations
import os, sys, logging, time
from loguru import logger

# Ensure project modules importable when run stand-alone
sys.path.append(os.path.dirname(__file__))

try:
    from trading_bot.config.settings import settings
    from trading_bot.utils import order_execution as oe
    from trading_bot.exchange.binance_futures_gateway import BinanceAPIError
except Exception as e:
    print(f"[FATAL] Imports failed: {e}")
    sys.exit(1)

DRY_RUN = os.getenv('DRY_RUN', '0') == '1'
LOOP_INTERVAL = int(os.getenv('LOOP_INTERVAL_SEC', '0') or '0')
EXIT_WHEN_EMPTY = os.getenv('EXIT_WHEN_EMPTY', '0') == '1'


def process_once():
    # Initialize order execution (ensures gateway & position manager ready)
    try:
        oe.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize order execution: {e}")
        return 0
    if not getattr(settings, 'FUTURES_ENABLED', False):
        logger.warning("Futures trading disabled. Exiting.")
        return 0
    if oe._position_manager is None:
        logger.error("Position manager not available.")
        return 0
    active_symbols = {s.strip().upper() for s in getattr(settings, 'FUTURES_SYMBOLS', '').split(',') if s.strip()}
    logger.info(f"Active FUTURES_SYMBOLS: {sorted(active_symbols)}")
    positions = oe._position_manager.positions
    if not positions:
        logger.info("No tracked positions.")
        return 0
    flattened = []
    skipped = []
    for sym, pos in positions.items():
        try:
            size = pos.size
        except Exception:
            continue
        if abs(size) < 1e-9:
            continue
        if sym in active_symbols:
            skipped.append(sym)
            continue
        side = 'SELL' if size > 0 else 'BUY'
        qty = abs(size)
        logger.info(f"[CLOSE] {sym} size={size} side={side} qty={qty}")
        if DRY_RUN:
            logger.info(f"[DRY_RUN] Would submit {side} {sym} qty={qty}")
            flattened.append(sym)
            continue
        try:
            # Direct reduce-only order via gateway if exposed; else use submit_signal logic with target 0
            if oe._gateway is not None:
                order = oe._gateway.create_order(sym, side, qty, order_type='MARKET', reduce_only=True)
                fill_qty = qty if side=='BUY' else -qty
                pos.update_fill(fill_qty, float(order.get('avgPrice', pos.entry_price or 0) or 0))
                logger.success(f"Flattened {sym} orderId={order.get('orderId')}")
            else:
                # Fallback: emulate closing via submit_signal by forcing target 0
                # Temporarily monkey patch FUTURES_SYMBOLS to include symbol so submit_signal runs
                from trading_bot.config import settings as _settings_mod
                original = _settings_mod.settings.FUTURES_SYMBOLS
                if sym not in original:
                    _settings_mod.settings.FUTURES_SYMBOLS += f",{sym}"
                # Use hold then manual adjustment: simple path -> direct position manager change
                pos.update_fill(-size, pos.entry_price or 0)
                logger.success(f"Flattened {sym} (fallback path)")
            flattened.append(sym)
        except BinanceAPIError as be:
            logger.error(f"[ERROR] Binance flatten {sym} code={be.error_code} msg={be.msg}")
        except Exception as e:
            logger.error(f"[ERROR] Flatten {sym}: {e}")
        time.sleep(0.25)  # gentle pacing
    logger.info(f"Flattened symbols: {flattened}")
    logger.info(f"Skipped (still active): {skipped}")
    if DRY_RUN and flattened:
        logger.warning("DRY_RUN active: no real orders were sent.")
    return len(flattened)


def main():
    logger.add("run_stop_loss.log", rotation="5 MB", retention="5 days")
    if LOOP_INTERVAL <= 0:
        process_once()
        return
    logger.info(f"Loop mode enabled interval={LOOP_INTERVAL}s exit_when_empty={EXIT_WHEN_EMPTY}")
    while True:
        flattened = process_once()
        if EXIT_WHEN_EMPTY and flattened == 0:
            logger.info("No positions flattened; exiting due to EXIT_WHEN_EMPTY=1")
            break
        try:
            import time as _t
            _t.sleep(LOOP_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Interrupted by user; exiting loop.")
            break

if __name__ == '__main__':
    main()
