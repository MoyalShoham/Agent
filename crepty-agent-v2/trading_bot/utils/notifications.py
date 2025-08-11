"""
Notification Hooks - Email/Telegram notifications for major events.
Extend with real API keys and logic as needed.
"""
from loguru import logger

def send_notification(message, channel='email'):
    logger.info(f"[Notification] ({channel}): {message}")
    # Placeholder: Integrate with real email/Telegram APIs
