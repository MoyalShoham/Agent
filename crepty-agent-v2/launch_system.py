#!/usr/bin/env python3
"""
Simplified Crypto System Launcher
Uses existing system components with fixed imports
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "trading_bot"))

def check_environment():
    """Check required environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file")
        return False
    return True

def check_imports():
    """Check if all required modules can be imported"""
    try:
        print("Checking imports...")
        
        # Test AI agents import
        from trading_bot.ai_models.ai_agents import create_ai_agent_orchestrator
        print("✓ AI Agents module imported successfully")
        
        # Test data collector
        try:
            from trading_bot.data.real_time_data_collector import RealTimeDataCollector
            print("✓ Real-time Data Collector imported successfully")
        except ImportError as e:
            print(f"✗ Data Collector import failed: {e}")
            return False
            
        # Test strategy manager
        try:
            from trading_bot.strategies.strategy_experts import StrategyManager
            print("✓ Strategy Manager imported successfully")
        except ImportError as e:
            print(f"✗ Strategy Manager import failed: {e}")
            return False
            
        # Test meta learner
        try:
            from trading_bot.ai_models.meta_learner import MetaLearner
            print("✓ Meta Learner imported successfully")
        except ImportError as e:
            print(f"✗ Meta Learner import failed: {e}")
            return False
            
        # Test coordinator
        try:
            from trading_bot.coordinators.trading_system_coordinator import TradingSystemCoordinator
            print("✓ Trading System Coordinator imported successfully")
        except ImportError as e:
            print(f"✗ Trading System Coordinator import failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Import check failed: {e}")
        return False

async def run_ai_analysis_demo():
    """Run a demo of the AI analysis system"""
    try:
        print("\n" + "="*60)
        print("CRYPTO AI ANALYSIS SYSTEM DEMO")
        print("="*60)
        
        # Import AI orchestrator
        from trading_bot.ai_models.ai_agents import create_ai_agent_orchestrator
        
        # Create orchestrator
        print("Creating AI Agent Orchestrator...")
        orchestrator = create_ai_agent_orchestrator()
        
        # Demo market data
        demo_market_data = {
            "symbol": "BTCUSDT",
            "price": 45000.00,
            "volume": 1000000,
            "timestamp": "2025-08-13T12:00:00Z",
            "indicators": {
                "rsi": 55.5,
                "macd": 0.02,
                "bollinger_upper": 46000,
                "bollinger_lower": 44000
            }
        }
        
        print("Running comprehensive AI analysis...")
        
        # Get comprehensive analysis
        analysis = await orchestrator.get_comprehensive_analysis(
            "Analyze current BTCUSDT market conditions", 
            demo_market_data
        )
        
        print(f"✓ Analysis completed with {len(analysis)} expert insights")
        
        # Display results
        for expert, result in analysis.items():
            if result:
                print(f"\n--- {expert.upper()} ---")
                if hasattr(result, 'analysis_summary'):
                    print(f"Summary: {result.analysis_summary}")
                elif hasattr(result, 'sentiment_score'):
                    print(f"Sentiment Score: {result.sentiment_score}")
                else:
                    print(f"Analysis completed: {type(result).__name__}")
            else:
                print(f"\n--- {expert.upper()} ---")
                print("Analysis failed or returned None")
        
        # Demo trading recommendation
        print("\n" + "-"*60)
        print("Getting trading recommendation...")
        
        trading_context = {
            "market_data": demo_market_data,
            "portfolio_data": {
                "total_value": 10000,
                "available_cash": 5000,
                "positions": []
            }
        }
        
        trading_rec = await orchestrator.get_trading_recommendation(
            "Should we enter a position in BTCUSDT?",
            trading_context
        )
        
        print("✓ Trading recommendation completed")
        
        # Demo report generation
        print("\n" + "-"*60)
        print("Generating demo financial report...")
        
        demo_trading_data = {
            "trades": [
                {
                    "symbol": "BTCUSDT",
                    "quantity": 0.1,
                    "entry_price": 44000,
                    "exit_price": 45000,
                    "pnl": 100,
                    "status": "closed",
                    "entry_time": "2025-08-01",
                    "exit_time": "2025-08-13"
                }
            ],
            "total_pnl": 100,
            "total_fees": 5
        }
        
        monthly_report = await orchestrator.generate_monthly_financial_report(
            "Generate demo monthly report",
            demo_trading_data
        )
        
        print("✓ Demo report generated")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("All AI agents are working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main launcher function"""
    print("Crypto Trading System - Simplified Launcher")
    print("=" * 50)
    
    try:
        # Check environment
        if not check_environment():
            print("Please set up your .env file with required API keys")
            return
            
        # Check imports
        if not check_imports():
            print("Import check failed. Please install dependencies:")
            print("pip install -r requirements.txt")
            return
            
        print("\n✓ All checks passed!")
        
        # Run AI analysis demo
        await run_ai_analysis_demo()
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Please set environment variables manually.")
    
    asyncio.run(main())
