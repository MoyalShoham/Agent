"""
Notification Hooks - Email/Telegram notifications for major events.
Extend with real API keys and logic as needed.
"""
from loguru import logger
from trading_bot.utils.event_bus import subscribe

def send_notification(message, channel='email'):
    logger.info(f"[Notification] ({channel}): {message}")
    # Placeholder: Integrate with real email/Telegram APIs

def _log_event(evt_type):
    def inner(payload):
        logger.info(f"EVENT {evt_type}: {payload}")
    return inner

for _evt in [
    'ORDER_SUBMITTED','ORDER_FILLED','ORDER_PARTIAL','ORDER_CANCELED','ORDER_REJECTED','POSITION_CHANGED'
]:
    try:
        subscribe(_evt, _log_event(_evt))
    except Exception:
        logger.exception('Failed subscribing to events')
