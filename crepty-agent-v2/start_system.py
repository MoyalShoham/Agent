#!/usr/bin/env python3
"""
Simple System Launcher - Starts the complete crypto trading system
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "trading_bot"))

from loguru import logger

async def main():
    """Main launcher"""
    try:
        # Check environment variables
        required_vars = ["OPENAI_API_KEY", "BINANCE_API_KEY", "BINANCE_SECRET_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            print(f"Missing environment variables: {', '.join(missing)}")
            print("Please set these in your .env file")
            return
            
        print("Starting Crypto Trading System...")
        print("=" * 50)
        
        # Import and run the main system
        from run_crypto_system import CryptoSystemLauncher
        
        launcher = CryptoSystemLauncher()
        await launcher.start_system()
        
    except KeyboardInterrupt:
        print("System stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
