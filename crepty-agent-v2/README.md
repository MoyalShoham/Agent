# Real-Time Crypto Trading Agent System

A modular, multi-agent cryptocurrency trading system in Python for real-time trading on Binance, with OpenAI-powered AI agents, robust error handling, and enterprise-grade reliability.

## Features
- Modular agent architecture (Manager, Research, Social, Analysis, Risk, Trader)
- Real-time data processing and trading
- OpenAI and Binance API integration
- Async/await concurrency
- Robust logging (loguru)
- Secure config via dotenv
- Pydantic data models
- Paper trading and backtesting
- Safety: daily loss/position limits, emergency stop

## Project Structure
```
trading_bot/
  agents/
  utils/
  config/
  tests/
main.py
requirements.txt
.env
```

## Setup
1. Create `.env` and fill in your API keys.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`

## Security
- Never commit real API keys.
- Use `.env` for secrets.
