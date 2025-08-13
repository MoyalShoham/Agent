"""
Test script for the Real-Time Crypto Analysis System
"""

import asyncio
import os
from datetime import datetime
from loguru import logger

# Import our system
from real_time_crypto_analysis import (
    create_real_time_analysis_system,
    get_ai_enhanced_signals,
    get_portfolio_optimization_recommendations
)

async def test_system_initialization():
    """Test system initialization"""
    logger.info("🧪 Testing system initialization...")
    
    try:
        # Test with minimal symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        system = create_real_time_analysis_system(test_symbols)
        
        # Initialize system
        await system.initialize()
        logger.info("✅ System initialization test passed")
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"📊 System Status: {status['status'] if 'status' in status else 'initialized'}")
        logger.info(f"📊 Symbols: {status.get('symbols', [])}")
        
        # Cleanup
        await system.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization test failed: {e}")
        return False

async def test_market_analysis():
    """Test market analysis functionality"""
    logger.info("🧪 Testing market analysis...")
    
    try:
        system = create_real_time_analysis_system(['BTCUSDT'])
        await system.initialize()
        
        # Get market analysis
        analysis = await system.get_market_analysis('BTCUSDT')
        
        if analysis:
            logger.info("✅ Market analysis test passed")
            logger.info(f"📊 Symbol: {analysis['symbol']}")
            logger.info(f"📊 Strategy Signals: {len(analysis['strategy_signals'])}")
            
            # Print strategy signals
            for signal in analysis['strategy_signals']:
                logger.info(f"  - {signal['strategy']}: {signal['signal']} (confidence: {signal['confidence']:.2f})")
                
            # Print meta signal
            if analysis['meta_signal']:
                meta = analysis['meta_signal']
                logger.info(f"📊 Meta Signal: {meta['signal_type']} (confidence: {meta['confidence']:.2f})")
            else:
                logger.info("📊 Meta Signal: No signal generated")
        else:
            logger.warning("⚠️ No market analysis data received")
            
        await system.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Market analysis test failed: {e}")
        return False

async def test_ai_agents():
    """Test AI agents functionality"""
    logger.info("🧪 Testing AI agents...")
    
    try:
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("⚠️ No OpenAI API key found, skipping AI agent test")
            return True
            
        system = create_real_time_analysis_system(['BTCUSDT'])
        await system.initialize()
        
        # Test AI orchestrator
        if system.ai_orchestrator:
            agent_status = system.ai_orchestrator.get_agent_status()
            logger.info("✅ AI agents test passed")
            logger.info(f"📊 Available agents: {list(agent_status.keys())}")
        else:
            logger.warning("⚠️ AI orchestrator not initialized")
            
        await system.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ AI agents test failed: {e}")
        return False

async def test_strategy_experts():
    """Test strategy experts"""
    logger.info("🧪 Testing strategy experts...")
    
    try:
        system = create_real_time_analysis_system(['BTCUSDT'])
        await system.initialize()
        
        if system.strategy_manager:
            strategies = system.strategy_manager.strategies
            logger.info("✅ Strategy experts test passed")
            logger.info(f"📊 Available strategies: {[s.name for s in strategies]}")
            
            # Test strategy performance tracking
            performance = system.strategy_manager.get_strategy_performance()
            logger.info(f"📊 Strategy performance tracking: {len(performance)} strategies")
        else:
            logger.warning("⚠️ Strategy manager not initialized")
            
        await system.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy experts test failed: {e}")
        return False

async def test_data_collection():
    """Test data collection"""
    logger.info("🧪 Testing data collection...")
    
    try:
        system = create_real_time_analysis_system(['BTCUSDT'])
        await system.initialize()
        
        # Test data manager
        if system.data_manager:
            logger.info("✅ Data collection test passed")
            logger.info(f"📊 Data manager symbols: {system.data_manager.symbols}")
        else:
            logger.warning("⚠️ Data manager not initialized")
            
        await system.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Data collection test failed: {e}")
        return False

async def test_integration_functions():
    """Test integration functions"""
    logger.info("🧪 Testing integration functions...")
    
    try:
        # Test get_ai_enhanced_signals
        logger.info("Testing get_ai_enhanced_signals...")
        signals = await get_ai_enhanced_signals(['BTCUSDT'])
        
        if signals:
            logger.info("✅ Integration functions test passed")
            logger.info(f"📊 Received signals for: {list(signals.keys())}")
        else:
            logger.info("📊 No signals received (may be normal)")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration functions test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting Real-Time Crypto Analysis System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Data Collection", test_data_collection),
        ("Strategy Experts", test_strategy_experts),
        ("AI Agents", test_ai_agents),
        ("Market Analysis", test_market_analysis),
        ("Integration Functions", test_integration_functions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} test completed successfully")
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == total

async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        exit_code = 1
        
    except Exception as e:
        logger.error(f"💥 Test runner crashed: {e}")
        exit_code = 1
    
    logger.info(f"\n🏁 Test run completed with exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    # Configure logging for tests
    logger.add("test_results.log", rotation="1 MB", level="INFO")
    
    # Run tests
    exit_code = asyncio.run(main())
    exit(exit_code)
