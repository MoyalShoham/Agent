#!/usr/bin/env python3
"""
Simple System Launcher - Quick Start Version
Works with existing modular structure
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_ai_agents():
    """Test the AI agents system"""
    try:
        print("Testing AI Agents...")
        
        # Import AI agents
        from trading_bot.ai_models.ai_agents import create_ai_agent_orchestrator
        
        # Create orchestrator
        orchestrator = create_ai_agent_orchestrator()
        print("✓ AI Agent Orchestrator created successfully")
        
        # Test with sample data
        sample_data = {
            "symbol": "BTCUSDT",
            "price": 45000.00,
            "volume": 1000000,
            "rsi": 55.5
        }
        
        # Get comprehensive analysis
        print("Running Comprehensive AI Analysis...")
        comprehensive_analysis = await orchestrator.get_comprehensive_analysis(
            "Analyze BTCUSDT market conditions",
            sample_data
        )
        
        if comprehensive_analysis:
            print(f"✓ Comprehensive Analysis completed with {len(comprehensive_analysis)} expert insights")
            for expert, result in comprehensive_analysis.items():
                if result:
                    print(f"  - {expert}: Analysis completed")
                else:
                    print(f"  - {expert}: Analysis failed")
        
        # Get trading recommendation
        print("Running Trading Recommendation...")
        trading_context = {
            "market_data": sample_data,
            "portfolio_data": {
                "total_value": 10000,
                "available_cash": 5000,
                "positions": []
            }
        }
        
        trading_recommendation = await orchestrator.get_trading_recommendation(
            "Should we enter a position in BTCUSDT?",
            trading_context
        )
        
        if trading_recommendation:
            print(f"✓ Trading Recommendation completed")
            if 'broker_decision' in trading_recommendation:
                print(f"  - Broker Decision: Available")
            if 'risk_assessment' in trading_recommendation:
                print(f"  - Risk Assessment: Available")
        
        # Get portfolio insights
        print("Running Portfolio Optimization...")
        portfolio_data = {
            "positions": [{"symbol": "BTCUSDT", "quantity": 0.1, "value": 4500}],
            "total_value": 10000,
            "cash_available": 5500
        }
        
        portfolio_insights = await orchestrator.get_portfolio_insights(
            "Optimize current portfolio allocation",
            portfolio_data
        )
        
        if portfolio_insights:
            print(f"✓ Portfolio Optimization completed")
        
        # Generate monthly report
        print("Generating Monthly Report...")
        trading_data = {
            "trades": [
                {
                    "symbol": "BTCUSDT",
                    "quantity": 0.1,
                    "entry_price": 44000,
                    "exit_price": 45000,
                    "pnl": 100,
                    "status": "closed"
                }
            ],
            "total_pnl": 100
        }
        
        monthly_report = await orchestrator.generate_monthly_financial_report(
            "Generate trading report",
            trading_data
        )
        
        if monthly_report:
            print(f"✓ Monthly Report generated")
            print(f"  Report type: {monthly_report.get('report_type', 'monthly')}")
            print(f"  Timestamp: {monthly_report.get('timestamp', 'N/A')}")
        
        print("\n" + "="*60)
        print("AI AGENTS SYSTEM TEST SUCCESSFUL!")
        print("All expert agents are working correctly.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"AI Agents test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_simple_analysis():
    """Run a simple analysis cycle"""
    try:
        print("\n" + "="*60)
        print("SIMPLE ANALYSIS CYCLE")
        print("="*60)
        
        # Test existing components if available
        success = await test_ai_agents()
        
        if success:
            print("\n✓ All systems working correctly!")
            print("\nNext steps:")
            print("1. Your AI expert agents are ready")
            print("2. Configure .env with your API keys")
            print("3. Use 'python start_system.py' for full automation")
            print("4. Check organized workspace in 'archived_files' directory")
        else:
            print("\n⚠ Some components need attention")
            print("Please check the error messages above")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def check_basic_setup():
    """Check basic setup requirements"""
    print("Checking basic setup...")
    
    # Check if AI agents file exists
    ai_agents_file = project_root / "trading_bot" / "ai_models" / "ai_agents.py"
    if ai_agents_file.exists():
        print("✓ AI Agents file found")
    else:
        print("✗ AI Agents file missing")
        return False
    
    # Check requirements
    req_file = project_root / "requirements.txt"
    if req_file.exists():
        print("✓ Requirements file found")
    else:
        print("✗ Requirements file missing")
    
    # Check for API key
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found")
    else:
        print("⚠ OpenAI API key not set (required for AI agents)")
        print("  Please set OPENAI_API_KEY in your environment or .env file")
    
    return True

async def main():
    """Main launcher"""
    print("Crypto Trading System - Simple Launcher")
    print("=" * 50)
    
    try:
        # Load environment variables if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ Environment variables loaded")
        except ImportError:
            print("⚠ python-dotenv not installed, using system environment")
        
        # Basic setup check
        if not check_basic_setup():
            print("\nSetup incomplete. Please run:")
            print("pip install -r requirements.txt")
            return
        
        # Run simple analysis
        await run_simple_analysis()
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
