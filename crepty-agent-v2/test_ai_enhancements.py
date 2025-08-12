"""
Test script for AI-Enhanced ML Signals
Run this to verify the new AI integration is working properly.
"""
import asyncio
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.utils.ai_enhanced_ml_signals import generate_ai_enhanced_ml_signal
from trading_bot.utils.ai_position_sizer import calculate_ai_position_size
from trading_bot.utils.binance_client import BinanceClient

async def test_ai_enhanced_signals():
    """Test the AI-enhanced ML signal generation"""
    print("🤖 Testing AI-Enhanced ML Signal Generation...")
    
    # Initialize Binance client
    binance = BinanceClient()
    
    # Test symbols from your active portfolio
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        try:
            # Get historical price data
            price_data = binance.get_ohlcv_dataframe(symbol, interval='1h', limit=100)
            
            if price_data.empty:
                print(f"  ❌ No price data available for {symbol}")
                continue
            
            # Generate AI-enhanced signal
            result = await generate_ai_enhanced_ml_signal(symbol, price_data)
            
            print(f"  🎯 Enhanced Signal: {result['enhanced_signal']}")
            print(f"  📈 Confidence: {result['confidence']:.2f}")
            print(f"  🤖 ML Signal: {result['ml_signal']}")
            print(f"  🧠 AI Enhancement: {result['ai_enhancement']}")
            print(f"  🌍 Market Regime: {result['market_regime']}")
            print(f"  💭 Reasoning: {result['reasoning'][:100]}...")
            print(f"  ⚠️  Risk Factors: {result['risk_factors']}")
            
            # Test AI position sizing
            if result['enhanced_signal'] != 0:
                current_price = float(price_data['close'].iloc[-1])
                available_capital = 10000.0  # Test with $10k
                
                portfolio_state = {
                    'positions': {},
                    'max_positions': 8,
                    'current_drawdown': 0.0
                }
                
                market_context = {
                    'market_regime': result['market_regime'],
                    'volatility': 'medium',
                    'correlation_risk': 'medium',
                    'liquidity': 'good'
                }
                
                position_result = await calculate_ai_position_size(
                    symbol=symbol,
                    signal_strength=result['enhanced_signal'],
                    signal_confidence=result['confidence'],
                    current_price=current_price,
                    available_capital=available_capital,
                    portfolio_state=portfolio_state,
                    market_context=market_context
                )
                
                print(f"  💰 AI Position Size: ${position_result['position_size_usd']:.2f} ({position_result['position_size_pct']:.1%})")
                print(f"  📊 Quantity: {position_result['quantity']:.6f}")
                print(f"  🛑 Stop Loss: {position_result['stop_loss_pct']:.1%}")
                print(f"  🎯 Take Profit: {position_result['take_profit_pct']:.1%}")
            
        except Exception as e:
            print(f"  ❌ Error testing {symbol}: {e}")

async def test_system_integration():
    """Test if all components are properly integrated"""
    print("\n🔧 Testing System Integration...")
    
    try:
        # Test imports
        from trading_bot.utils.openai_client import OpenAIClient
        print("  ✅ OpenAI client import successful")
        
        # Test OpenAI connection
        openai_client = OpenAIClient()
        test_response = await openai_client.ask("Test message: respond with 'AI integration working'")
        if test_response and "working" in test_response.lower():
            print("  ✅ OpenAI API connection successful")
        else:
            print("  ⚠️ OpenAI API response unclear")
        
        # Test ML model availability
        from trading_bot.utils.ml_signals import _ml_generator
        if _ml_generator.is_trained:
            print("  ✅ ML model loaded and ready")
        else:
            print("  ⚠️ ML model not trained (will use default signals)")
        
        # Test environment variables
        import os
        if os.getenv('AI_ENHANCED_ML_ENABLED') == 'true':
            print("  ✅ AI enhancements enabled in configuration")
        else:
            print("  ⚠️ AI enhancements not enabled in .env")
        
        print("\n🎉 System integration test completed!")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")

async def main():
    """Main test function"""
    print("🚀 AI-Enhanced Trading System Test")
    print("=" * 50)
    
    await test_system_integration()
    await test_ai_enhanced_signals()
    
    print("\n" + "=" * 50)
    print("✅ Test completed! Check the results above.")
    print("📝 If all tests pass, your AI enhancements are ready!")

if __name__ == "__main__":
    asyncio.run(main())
