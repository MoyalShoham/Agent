# HOW TO RUN THE CRYPTO TRADING SYSTEM

Great! Your workspace has been organized successfully. Here's how to run the complete system:

## WORKSPACE SUMMARY
- Moved 21 log files to archive
- Moved 36 deprecated files to archive  
- Moved 17 data files to data directory
- Organized 10 scripts into categories
- Organized 5 test files
- Cleaned up 1426 __pycache__ directories

## STEP 1: SETUP ENVIRONMENT

### 1.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.2 Configure Environment Variables
Edit the `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=false
```

### 1.3 Review Configuration
Edit `config.json` to customize:
- Trading symbols
- Position sizes
- Risk parameters
- AI agent settings
- Report preferences

## STEP 2: RUN THE SYSTEM

### Option A: Simple Launcher (Recommended)
```bash
python start_system.py
```

### Option B: Full System Launcher
```bash
python run_crypto_system.py
```

### Option C: Windows Batch File
```bash
start_crypto_system.bat
```

### Option D: Unix/Mac Shell Script
```bash
./start_crypto_system.sh
```

## STEP 3: SYSTEM COMPONENTS

### AI EXPERT AGENTS (8 Total)
1. **CryptoCurrency Analysis Expert** - On-chain analysis, DeFi protocols
2. **Financial Expert** - Macroeconomic analysis, risk management
3. **Expert Broker** - Trade execution, order optimization
4. **Risk Management Expert** - VaR, stress testing, portfolio risk
5. **Market Sentiment Expert** - Social sentiment, news analysis
6. **Portfolio Optimization Expert** - Asset allocation, rebalancing
7. **Technical Analysis Expert** - Chart patterns, indicators
8. **Accountant Expert** - Financial reports, tax documents

### TRADING STRATEGIES (4 Total)
1. **Trend Following** - Momentum-based trading
2. **Volatility Breakout** - Breakout pattern trading
3. **Funding Squeeze** - Funding rate arbitrage
4. **Mean Reversion** - Counter-trend trading

### DATA COLLECTION
- Real-time Binance WebSocket streams
- Historical OHLCV data
- Order book data
- Funding rates and liquidations

### AUTOMATED FEATURES
- **Real-time Analysis**: Every 5 minutes
- **Comprehensive Analysis**: Every 15 minutes
- **Monthly Reports**: Auto-generated on 1st of month
- **Yearly Reports**: Auto-generated on January 1st
- **Tax Reports**: Form 8949 compatible

## STEP 4: MONITOR THE SYSTEM

### Logs Location
- Main logs: `/logs/crypto_system_YYYY-MM-DD.log`
- Error logs: `/logs/crypto_system_errors_YYYY-MM-DD.log`
- Trading logs: `/logs/`

### Reports Location
- Monthly: `/reports/monthly/`
- Yearly: `/reports/yearly/`
- Tax: `/reports/tax/`

### Real-time Monitoring
The system provides console output with:
- System status updates
- Trading decisions
- AI analysis results
- Risk assessments
- Performance metrics

## STEP 5: CONFIGURATION OPTIONS

### Trading Configuration
```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "position_size": 0.1,
    "max_positions": 5,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04
  }
}
```

### AI Agents Configuration
```json
{
  "ai_agents": {
    "enabled": true,
    "analysis_interval": 300,
    "comprehensive_analysis_interval": 900,
    "model": "gpt-4o"
  }
}
```

### Risk Management
```json
{
  "risk_management": {
    "max_portfolio_risk": 0.05,
    "max_single_position_risk": 0.02,
    "emergency_stop_loss": 0.10
  }
}
```

## STEP 6: ACCESSING REPORTS

### Manual Report Generation
```python
# Generate monthly report
orchestrator = create_ai_agent_orchestrator()
report = await orchestrator.generate_monthly_financial_report(
    "Generate monthly report", trading_data
)

# Generate yearly report
yearly_report = await orchestrator.generate_yearly_financial_report(
    "Generate yearly report", trading_data
)

# Generate tax reports
tax_reports = await orchestrator.generate_tax_reports(trading_data)
```

### Report Files Generated
- `monthly_report_YYYYMMDD_HHMMSS.csv`
- `monthly_report_YYYYMMDD_HHMMSS_positions.csv`
- `monthly_report_YYYYMMDD_HHMMSS_tax_summary.csv`
- `form_8949_crypto_YYYYMMDD_HHMMSS.csv`

## STEP 7: TROUBLESHOOTING

### Common Issues
1. **Missing API Keys**: Check `.env` file
2. **Network Issues**: Check internet connection
3. **Import Errors**: Run `pip install -r requirements.txt`
4. **Permission Errors**: Run as administrator if needed

### System Status Check
```python
from run_crypto_system import CryptoSystemLauncher
launcher = CryptoSystemLauncher()
status = launcher.get_system_status()
print(json.dumps(status, indent=2))
```

## STEP 8: STOPPING THE SYSTEM

- **Ctrl+C**: Graceful shutdown
- **Emergency Stop**: Close terminal/command prompt

## FOLDER STRUCTURE

```
crepty-agent-v2/
├── trading_bot/           # Core trading system
│   ├── ai_models/         # AI agents and orchestrator
│   ├── strategies/        # Trading strategies
│   ├── coordinators/      # System coordination
│   └── ...
├── data/                  # Market data and models
├── reports/               # Generated reports
├── logs/                  # System logs
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── archive/               # Old/deprecated files
├── config.json            # System configuration
├── .env                   # Environment variables
├── run_crypto_system.py   # Main system launcher
└── start_system.py        # Simple launcher
```

## READY TO START!

Your system is now organized and ready to run. Execute:

```bash
python start_system.py
```

The system will automatically:
- Initialize all 8 AI expert agents
- Start real-time data collection
- Begin trading with 4 strategies
- Generate reports monthly/yearly
- Provide comprehensive market analysis

Enjoy your automated crypto trading system!
