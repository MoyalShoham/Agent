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
- **Regime-adaptive strategy selection** (bull, bear, sideways)
- **Auto-optimization**: strategies are periodically re-tuned using latest data
- **Live parameter reloading**: system always uses best parameters, no restart needed
- **Performance visualization**: plot strategy PnL and trades with `visualize_analytics.py`

## Project Structure
```
trading_bot/
  agents/
  utils/
  config/
  strategies/
  tests/
main.py
requirements.txt
.env
```
## Advanced Features

- **Strategy Ensemble**: Includes momentum, mean reversion, breakout, MA crossover, volatility expansion, RSI, MACD, Bollinger Bands, and more. Strategies are weighted and selected in real time based on recent performance and market regime.
- **Auto-Optimizer**: `auto_optimizer.py` periodically finds the best parameters for each strategy and updates the live system.
- **Visualization**: Run `python trading_bot/utils/visualize_analytics.py` to compare strategy performance and portfolio analytics.


## Setup
1. Create `.env` and fill in your API keys.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`

## Security
- Never commit real API keys.
- Use `.env` for secrets.
