# Real-Time Crypto Analysis System

A comprehensive, production-grade crypto trading system that integrates real-time data APIs, advanced trading strategies, AI agents, and meta-learning for optimal trading decisions.

## 🚀 Features

### Real-Time Data Collection
- **Binance Futures WebSocket**: Live trades, order book, mark price, liquidation streams
- **Binance REST API**: Historical klines, funding rates, open interest
- **Multiple Aggregators**: CoinGecko, CryptoCompare support
- **Data Storage**: SQLite-based historical data storage with efficient querying

### Strategy Experts
- **Trend Following**: Donchian breakouts with HTF trend filters
- **Volatility Breakout**: Compression to expansion detection
- **Funding Squeeze**: Extreme funding rate reversal strategies  
- **Mean Reversion**: VWAP-based range trading with spread filtering

### AI Agents (Agentic Method)
- **Financial Analysis Agent**: Technical and fundamental market analysis
- **Broker Agent**: Trade execution and order management decisions
- **Risk Management Agent**: Portfolio risk assessment and control
- **Market Sentiment Agent**: Social sentiment and news impact analysis
- **Portfolio Optimization Agent**: Asset allocation and rebalancing

### Meta-Learning System
- **Contextual Bandits**: Strategy allocation based on market conditions
- **Ensemble Models**: LightGBM, Random Forest, Gradient Boosting
- **Regime Detection**: Volatility, trend, and sentiment regime classification
- **Performance Tracking**: Real-time strategy and model performance monitoring

## 📁 Architecture

```
trading_bot/
├── data/
│   └── real_time_data_collector.py     # WebSocket & REST data collection
├── strategies/
│   └── strategy_experts.py             # Individual trading strategies
├── ai_models/
│   ├── ai_agents.py                    # AI agents with OpenAI integration
│   └── meta_learner.py                 # Meta-learning and optimization
├── coordinators/
│   └── trading_system_coordinator.py   # Main system coordinator
└── real_time_crypto_analysis.py        # Main integration interface
```

## 🛠️ Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Environment Configuration**:
Ensure your `.env` file contains:
```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
BINANCE_FUTURES_API_KEY=your_futures_api_key
BINANCE_FUTURES_API_SECRET=your_futures_secret

# Trading Configuration
FUTURES_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT
MAX_CONCURRENT_POSITIONS=7
RISK_PER_TRADE=0.004
AI_CONFIDENCE_THRESHOLD=0.45
```

## 🎯 Usage

### Standalone Execution

```python
import asyncio
from real_time_crypto_analysis import create_real_time_analysis_system

async def main():
    # Create system with custom symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    system = create_real_time_analysis_system(symbols)
    
    # Start the complete trading system
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with Existing System

```python
from real_time_crypto_analysis import (
    get_ai_enhanced_signals,
    get_portfolio_optimization_recommendations,
    emergency_risk_check
)

# Get AI-enhanced signals for specific symbols
signals = await get_ai_enhanced_signals(['BTCUSDT', 'ETHUSDT'])

# Get portfolio optimization recommendations
portfolio_analysis = await get_portfolio_optimization_recommendations()

# Perform emergency risk assessment
risk_assessment = await emergency_risk_check()
```

### Manual Analysis

```python
from real_time_crypto_analysis import RealTimeCryptoAnalysisSystem

system = RealTimeCryptoAnalysisSystem(['BTCUSDT'])
await system.initialize()

# Get comprehensive market analysis
analysis = await system.get_market_analysis('BTCUSDT')
print(f"Meta Signal: {analysis['meta_signal']['signal_type']}")
print(f"Confidence: {analysis['meta_signal']['confidence']:.2f}")

# Get portfolio analysis
portfolio = await system.get_portfolio_analysis()
print(f"Active Positions: {len(portfolio['portfolio_data']['current_positions'])}")
```

## 🤖 AI Agent System

The system uses OpenAI's API with structured outputs and the agentic method:

### Financial Analysis Agent
- Provides technical and fundamental analysis
- Identifies market regimes and key levels
- Returns structured `FinancialAnalysisResponse`

### Broker Agent  
- Makes trading decisions and execution plans
- Considers liquidity, timing, and risk management
- Returns structured `BrokerResponse`

### Risk Management Agent
- Assesses portfolio risk and exposure
- Provides position sizing and stop-loss recommendations
- Returns structured `RiskManagementResponse`

### Example AI Agent Usage:

```python
# The agents automatically use this structure:
parser = PydanticOutputParser(pydantic_object=FinancialAnalysisResponse)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert financial analyst..."),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(llm=gpt, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

## 📊 Strategy Experts

### Trend Following Strategy
- Uses Donchian channel breakouts
- HTF EMA trend filter (50/200)
- Volume confirmation required
- ATR-based position sizing

### Volatility Breakout Strategy
- Detects BB width compression
- Realized volatility percentile analysis
- Volume surge confirmation
- Expansion breakout entries

### Funding Squeeze Strategy
- Monitors extreme funding rates (>1%)
- Open interest change analysis
- CVD divergence detection
- Reversal trade opportunities

### Mean Reversion Strategy
- VWAP-based z-score analysis
- Range market detection
- Spread filtering (max 10 bps)
- Tight risk management

## 🧠 Meta-Learning System

### Contextual Bandit
- Strategy selection based on market context
- Upper Confidence Bound algorithm
- Dynamic strategy weight adjustment
- Performance-based allocation

### Feature Engineering
- Market microstructure features
- Strategy signal agreement
- AI agent insights integration
- Time-based and regime features

### Model Training
- Walk-forward validation
- Time series cross-validation
- Ensemble of LightGBM/RF/GB models
- Automatic retraining every 1000 samples

## 📈 Performance Monitoring

### Real-Time Metrics
- Win rate and PnL tracking
- Sharpe ratio calculation
- Drawdown monitoring
- Strategy attribution

### Meta-Learning Performance
- Signal accuracy tracking
- Confidence calibration
- Strategy allocation efficiency
- Model drift detection

## ⚠️ Risk Management

### Position-Level Risk
- ATR-based position sizing
- Dynamic stop losses
- Volatility regime adjustment
- Correlation monitoring

### Portfolio-Level Risk
- Maximum concurrent positions
- Daily loss limits
- Emergency position closing
- Regime-based exposure limits

### AI-Enhanced Risk
- Real-time risk scoring
- Predictive risk models
- Emergency action recommendations
- Automated risk alerts

## 🔧 Configuration

### Key Parameters

```python
config = {
    'signal_cooldown_minutes': 3,           # Minimum time between signals
    'max_concurrent_positions': 7,          # Maximum open positions
    'min_confidence_threshold': 0.45,       # Minimum AI confidence
    'position_update_interval': 30,         # Position monitoring frequency
    'risk_per_trade': 0.004,               # Risk per trade (0.4%)
    'max_position_size': 0.12,             # Maximum position size (12%)
    'max_daily_losses': 0.05               # Daily loss limit (5%)
}
```

### Strategy Parameters

Each strategy expert has customizable parameters:

```python
# Trend Following
trend_params = {
    'donchian_period': 20,
    'htf_ema_fast': 50,
    'htf_ema_slow': 200,
    'atr_multiplier': 2.0,
    'min_breakout_volume': 1.5
}

# Volatility Breakout  
vol_params = {
    'bb_period': 20,
    'rv_percentile_low': 20,
    'min_expansion_factor': 1.5,
    'volume_surge_factor': 2.0
}
```

## 🚨 Emergency Features

### Emergency Risk Assessment
```python
assessment = await system.emergency_risk_assessment()
if assessment['emergency_assessment'].overall_risk_level == 'extreme':
    # Take emergency actions
    for action in assessment['emergency_assessment'].emergency_actions:
        print(f"Emergency: {action}")
```

### Automatic Shutdown Conditions
- Daily loss limit exceeded
- Extreme market volatility
- API connection failures
- Risk management failures

## 📝 Logging

The system provides comprehensive logging:

```
🚀 Starting Real-Time Crypto Trading System...
✅ Data manager initialized
✅ Strategy manager initialized with 4 strategies
✅ AI agent orchestrator initialized
✅ Meta-learner initialized
🎯 All components initialized successfully
📊 Performance Update:
  Total Equity: $10,000.00
  Active Positions: 2
  Win Rate: 65.23%
  Total PnL: 12.45%
  Meta-learner Win Rate: 68.12%
```

## 🤝 Integration with Existing System

The system is designed to integrate seamlessly with your existing trading bot:

1. **Import the analysis functions** into your current main.py
2. **Use the AI-enhanced signals** alongside your existing strategies
3. **Leverage the risk management** for better capital protection
4. **Monitor performance** through the integrated metrics

Example integration:

```python
# In your existing main.py
from real_time_crypto_analysis import get_ai_enhanced_signals

async def enhanced_trading_loop():
    # Your existing logic
    
    # Add AI-enhanced analysis
    ai_signals = await get_ai_enhanced_signals(your_symbols)
    
    for symbol, analysis in ai_signals.items():
        meta_signal = analysis['meta_signal']
        if meta_signal['confidence'] > 0.7:
            # Execute trade based on AI recommendation
            await execute_trade(symbol, meta_signal)
```

## 🔮 Future Enhancements

- **On-chain Analysis**: Integration with Glassnode/Santiment
- **Cross-Exchange Arbitrage**: Multi-exchange strategy support  
- **Advanced ML Models**: Transformer-based sequence models
- **Social Sentiment**: Twitter/Reddit sentiment integration
- **News Analysis**: Real-time news impact assessment
- **Options Strategies**: Crypto options trading strategies

## 📞 Support

For questions or issues:
1. Check the logs for detailed error information
2. Verify API keys and permissions
3. Ensure sufficient account balance
4. Monitor rate limits and connection stability

The system is designed to be robust and self-healing, with comprehensive error handling and automatic recovery mechanisms.
