#!/usr/bin/env python3
"""
Quick Implementation Script - Apply immediate improvements to your trading agent.
Run this script to quickly implement the most critical enhancements.
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_enhanced_components():
    """Setup enhanced components for immediate use"""
    logger.info("Setting up enhanced trading components...")
    
    # 1. Initialize Enhanced ML Signals
    try:
        from trading_bot.utils.enhanced_ml_signals import MLSignalGenerator, train_ml_model
        
        # Check if we have historical data to train on
        if os.path.exists('BTCUSDT_1h.csv'):
            logger.info("Training ML model with historical data...")
            
            # Load historical data
            df = pd.read_csv('BTCUSDT_1h.csv')
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Train the model
            ml_generator = MLSignalGenerator()
            results = ml_generator.train_model(df)
            logger.info(f"ML model training completed: {results}")
        else:
            logger.warning("No historical data found for ML training. Please download some first.")
            
    except Exception as e:
        logger.error(f"Error setting up ML signals: {e}")
    
    # 2. Initialize Enhanced Risk Manager
    try:
        from trading_bot.risk.enhanced_risk_manager import EnhancedRiskManager
        
        risk_manager = EnhancedRiskManager(
            max_portfolio_var=0.02,
            max_position_size=0.1,
            max_correlation=0.7,
            max_concentration=0.25
        )
        logger.info("Enhanced risk manager initialized")
        
        # Test with some dummy positions
        risk_manager.update_position('BTCUSDT', 1.0, 45000, 46000)
        risk_manager.update_position('ETHUSDT', 10.0, 3000, 3100)
        
        metrics = risk_manager.get_risk_metrics()
        logger.info(f"Risk metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error setting up enhanced risk manager: {e}")
    
    # 3. Setup WebSocket Client (optional - for real-time data)
    try:
        from trading_bot.utils.websocket_client import BinanceWebSocketClient
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        ws_client = BinanceWebSocketClient(symbols)
        
        # Add a simple price callback
        def on_price_update(symbol, price):
            logger.info(f"Price update: {symbol} = ${price:,.2f}")
        
        ws_client.add_price_callback(on_price_update)
        logger.info("WebSocket client setup complete (not started)")
        
    except Exception as e:
        logger.error(f"Error setting up WebSocket client: {e}")
    
    # 4. Setup Portfolio Optimizer
    try:
        from trading_bot.utils.advanced_portfolio_optimizer import AdvancedPortfolioOptimizer
        
        optimizer = AdvancedPortfolioOptimizer()
        logger.info("Portfolio optimizer initialized")
        
        # Test with dummy returns data
        if os.path.exists('BTCUSDT_1h.csv'):
            df = pd.read_csv('BTCUSDT_1h.csv')
            if 'close' in df.columns and len(df) > 100:
                # Create returns data for multiple assets (simulation)
                returns_data = pd.DataFrame({
                    'BTCUSDT': df['close'].pct_change().dropna(),
                    'ETHUSDT': (df['close'] * 0.8).pct_change().dropna(),  # Simulated ETH
                    'BNBUSDT': (df['close'] * 0.4).pct_change().dropna()   # Simulated BNB
                })
                
                weights, metrics = optimizer.optimize_portfolio(returns_data)
                logger.info(f"Portfolio optimization test - Weights: {weights}")
                logger.info(f"Portfolio metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error setting up portfolio optimizer: {e}")

def update_requirements():
    """Update requirements.txt with new dependencies"""
    additional_requirements = [
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "websockets>=10.0",
        "joblib>=1.0.0"
    ]
    
    try:
        # Read existing requirements
        with open('requirements.txt', 'r') as f:
            existing = f.read().strip().split('\n')
        
        # Add new requirements if not already present
        updated = existing.copy()
        for req in additional_requirements:
            package_name = req.split('>=')[0].split('==')[0]
            if not any(package_name in line for line in existing):
                updated.append(req)
        
        # Write updated requirements
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(updated))
        
        logger.info("Requirements.txt updated with new dependencies")
        logger.info("Run: pip install -r requirements.txt")
        
    except Exception as e:
        logger.error(f"Error updating requirements: {e}")

def create_example_config():
    """Create example configuration for enhanced features"""
    config_additions = """
# Enhanced Trading Agent Configuration
# Add these to your .env file:

# ML Model Settings
BYPASS_META=1
ML_MODEL_PATH=ml_model.pkl
ML_CONFIDENCE_THRESHOLD=0.6
ML_RETRAIN_INTERVAL=86400  # 24 hours

# Enhanced Risk Management
MAX_PORTFOLIO_VAR=0.02
MAX_POSITION_SIZE=0.1
MAX_CORRELATION=0.7
MAX_CONCENTRATION=0.25
PORTFOLIO_REBALANCE_THRESHOLD=0.05

# WebSocket Settings
WEBSOCKET_ENABLED=1
WEBSOCKET_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT

# Portfolio Optimization
PORTFOLIO_OPTIMIZATION_METHOD=max_sharpe
PORTFOLIO_REBALANCE_FREQUENCY=21  # days

# Advanced Features
REGIME_DETECTION_ENABLED=1
ALTERNATIVE_DATA_ENABLED=0
SENTIMENT_ANALYSIS_ENABLED=0
"""
    
    with open('enhanced_config_example.txt', 'w') as f:
        f.write(config_additions)
    
    logger.info("Example enhanced configuration saved to enhanced_config_example.txt")

def backup_current_ml_signals():
    """Backup current ML signals file before replacing"""
    ml_signals_path = 'trading_bot/utils/ml_signals.py'
    if os.path.exists(ml_signals_path):
        backup_path = f'trading_bot/utils/ml_signals_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
        os.rename(ml_signals_path, backup_path)
        logger.info(f"Backed up original ml_signals.py to {backup_path}")

def integrate_enhanced_ml_signals():
    """Replace the placeholder ML signals with enhanced version"""
    try:
        # Create symlink or copy enhanced ML signals
        enhanced_path = 'trading_bot/utils/enhanced_ml_signals.py'
        target_path = 'trading_bot/utils/ml_signals.py'
        
        if os.path.exists(enhanced_path):
            # Backup original
            backup_current_ml_signals()
            
            # Copy enhanced version
            import shutil
            shutil.copy2(enhanced_path, target_path)
            logger.info("Enhanced ML signals integrated successfully")
        else:
            logger.warning("Enhanced ML signals file not found")
            
    except Exception as e:
        logger.error(f"Error integrating enhanced ML signals: {e}")

def run_quick_setup():
    """Run the complete quick setup"""
    logger.info("🚀 Starting Quick Setup for Enhanced Trading Agent")
    
    # Step 1: Update requirements
    logger.info("📦 Updating requirements...")
    update_requirements()
    
    # Step 2: Create configuration example
    logger.info("⚙️ Creating configuration examples...")
    create_example_config()
    
    # Step 3: Integrate enhanced ML signals
    logger.info("🤖 Integrating enhanced ML signals...")
    integrate_enhanced_ml_signals()
    
    # Step 4: Setup enhanced components
    logger.info("🔧 Setting up enhanced components...")
    setup_enhanced_components()
    
    logger.info("✅ Quick setup completed!")
    print("\n" + "="*60)
    print("🎉 ENHANCED TRADING AGENT SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Review enhanced_config_example.txt")
    print("3. Add desired settings to your .env file")
    print("4. Test the enhanced features:")
    print("   python -c \"from trading_bot.utils.enhanced_ml_signals import generate_ml_signal; print('ML ready!')\"")
    print("5. Start your trading agent: python main.py")
    print("\n💡 Check IMPROVEMENT_RECOMMENDATIONS.md for detailed enhancement guide")

if __name__ == "__main__":
    run_quick_setup()
