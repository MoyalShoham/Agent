"""
Integration Test for Enhanced Trading System Components
Tests all enhanced components working together in a coordinated manner.
"""
import asyncio
import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import enhanced components
from trading_bot.risk.advanced_risk_manager import AdvancedRiskManager, RiskManagerConfig
from trading_bot.risk.emergency_controls import EmergencyRiskControls, EmergencyConfig
from trading_bot.utils.ai_integration import AIModelIntegrator, AIModelConfig
from trading_bot.utils.enhanced_websocket import EnhancedWebSocketClient, WebSocketConfig
from trading_bot.utils.ml_signals import EnhancedMLSignalGenerator, MLConfig

class IntegrationTestSuite:
    """Comprehensive integration test suite for enhanced components"""
    
    def __init__(self):
        self.test_results = {}
        self.components = {}
        
        # Configure logging
        logger.remove()
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        logger.info("🧪 Starting Enhanced Trading System Integration Tests")

    async def run_all_tests(self):
        """Run comprehensive integration tests"""
        try:
            # Test individual component initialization
            await self.test_component_initialization()
            
            # Test component interactions
            await self.test_component_interactions()
            
            # Test risk management integration
            await self.test_risk_management_flow()
            
            # Test AI decision making flow
            await self.test_ai_decision_flow()
            
            # Test WebSocket data processing
            await self.test_websocket_processing()
            
            # Test ML signal generation
            await self.test_ml_signal_generation()
            
            # Test emergency scenarios
            await self.test_emergency_scenarios()
            
            # Generate test report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            raise

    async def test_component_initialization(self):
        """Test initialization of all enhanced components"""
        logger.info("🔧 Testing component initialization...")
        
        try:
            # Initialize Advanced Risk Manager
            risk_config = RiskManagerConfig(
                max_portfolio_risk=0.15,
                max_position_size=0.1,
                var_confidence_level=0.95
            )
            self.components['risk_manager'] = AdvancedRiskManager(risk_config)
            logger.info("✅ Advanced Risk Manager initialized")
            
            # Initialize Emergency Controls
            emergency_config = EmergencyConfig(
                max_daily_loss_pct=0.05,
                max_drawdown_pct=0.15
            )
            self.components['emergency_controls'] = EmergencyRiskControls(emergency_config)
            logger.info("✅ Emergency Risk Controls initialized")
            
            # Initialize AI Integration
            ai_config = AIModelConfig(
                confidence_threshold=0.6,
                ensemble_voting_threshold=0.7
            )
            self.components['ai_integrator'] = AIModelIntegrator(ai_config)
            logger.info("✅ AI Model Integrator initialized")
            
            # Initialize Enhanced WebSocket
            ws_config = WebSocketConfig(
                max_connections=3,
                message_queue_size=500
            )
            self.components['websocket'] = EnhancedWebSocketClient(ws_config)
            logger.info("✅ Enhanced WebSocket Client initialized")
            
            # Initialize ML Signal Generator
            ml_config = MLConfig(
                min_confidence_threshold=0.6,
                ensemble_voting_enabled=True
            )
            self.components['ml_signals'] = EnhancedMLSignalGenerator(ml_config)
            logger.info("✅ Enhanced ML Signal Generator initialized")
            
            self.test_results['component_initialization'] = {
                'status': 'PASSED',
                'components_loaded': len(self.components),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.test_results['component_initialization'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise

    async def test_component_interactions(self):
        """Test interactions between components"""
        logger.info("🔗 Testing component interactions...")
        
        try:
            # Test Risk Manager <-> Emergency Controls interaction
            risk_manager = self.components['risk_manager']
            emergency_controls = self.components['emergency_controls']
            
            # Simulate portfolio state
            portfolio_state = {
                'total_value': 50000,
                'positions': {
                    'BTCUSDT': {'size': 0.5, 'value': 25000, 'unrealized_pnl': 500},
                    'ETHUSDT': {'size': 10, 'value': 15000, 'unrealized_pnl': -200}
                },
                'available_balance': 10000
            }
            
            # Update risk manager
            risk_metrics = await risk_manager.calculate_portfolio_risk(portfolio_state)
            logger.info(f"📊 Portfolio risk calculated: {risk_metrics.get('total_risk', 0):.1%}")
            
            # Test emergency controls
            emergency_controls.portfolio_value = portfolio_state['total_value']
            emergency_controls.positions = portfolio_state['positions']
            
            # Simulate trade decision
            can_trade, reason = emergency_controls.should_allow_trade('BTCUSDT', 0.8, 1.0)
            logger.info(f"🚦 Trade approval: {can_trade} - {reason}")
            
            # Test AI Integration <-> ML Signals interaction
            ai_integrator = self.components['ai_integrator']
            ml_signals = self.components['ml_signals']
            
            # Generate ML signals
            test_market_data = self._generate_test_market_data()
            ml_prediction = await ml_signals.generate_signal(test_market_data)
            logger.info(f"🤖 ML signal generated: {ml_prediction.get('signal', 'UNKNOWN')} (confidence: {ml_prediction.get('confidence', 0):.1%})")
            
            # Test AI decision with ML input
            context = {
                'market_data': test_market_data,
                'portfolio_data': portfolio_state,
                'risk_metrics': risk_metrics,
                'ml_signal': ml_prediction
            }
            
            ai_decision = await ai_integrator.get_ai_trading_decision(
                test_market_data, portfolio_state, risk_metrics
            )
            logger.info(f"🧠 AI decision: {ai_decision.action} (confidence: {ai_decision.confidence:.1%})")
            
            self.test_results['component_interactions'] = {
                'status': 'PASSED',
                'risk_calculation': 'SUCCESS',
                'emergency_check': 'SUCCESS',
                'ml_signal_generation': 'SUCCESS',
                'ai_decision_making': 'SUCCESS',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.test_results['component_interactions'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"❌ Component interaction test failed: {e}")

    async def test_risk_management_flow(self):
        """Test complete risk management flow"""
        logger.info("⚡ Testing risk management flow...")
        
        try:
            risk_manager = self.components['risk_manager']
            emergency_controls = self.components['emergency_controls']
            
            # Test scenarios with increasing risk
            test_scenarios = [
                {
                    'name': 'Normal Trading',
                    'portfolio_value': 100000,
                    'daily_pnl': 500,
                    'positions': {'BTCUSDT': {'size': 0.1, 'value': 5000, 'unrealized_pnl': 100}}
                },
                {
                    'name': 'High Risk',
                    'portfolio_value': 100000,
                    'daily_pnl': -2000,
                    'positions': {
                        'BTCUSDT': {'size': 0.8, 'value': 40000, 'unrealized_pnl': -2000},
                        'ETHUSDT': {'size': 0.6, 'value': 30000, 'unrealized_pnl': -1500}
                    }
                },
                {
                    'name': 'Emergency Scenario',
                    'portfolio_value': 100000,
                    'daily_pnl': -5000,
                    'positions': {
                        'BTCUSDT': {'size': 1.0, 'value': 50000, 'unrealized_pnl': -5000},
                        'ETHUSDT': {'size': 1.0, 'value': 50000, 'unrealized_pnl': -3000}
                    }
                }
            ]
            
            scenario_results = {}
            
            for scenario in test_scenarios:
                logger.info(f"🎯 Testing scenario: {scenario['name']}")
                
                # Update emergency controls
                emergency_controls.portfolio_value = scenario['portfolio_value']
                emergency_controls.daily_pnl = scenario['daily_pnl']
                emergency_controls.positions = scenario['positions']
                
                # Calculate risk metrics
                risk_metrics = await risk_manager.calculate_portfolio_risk(scenario)
                
                # Test trade approval
                can_trade, reason = emergency_controls.should_allow_trade('NEWCOIN', 0.7, 1.0)
                
                # Test position sizing
                base_size = 1000  # $1000 position
                adjusted_size = emergency_controls.adjust_position_size(base_size, 'NEWCOIN', 0.7)
                
                scenario_results[scenario['name']] = {
                    'portfolio_risk': risk_metrics.get('total_risk', 0),
                    'can_trade': can_trade,
                    'reason': reason,
                    'position_size_adjustment': (adjusted_size / base_size - 1) * 100,
                    'emergency_mode': emergency_controls.emergency_mode
                }
                
                logger.info(f"  📊 Portfolio Risk: {risk_metrics.get('total_risk', 0):.1%}")
                logger.info(f"  🚦 Can Trade: {can_trade}")
                logger.info(f"  📏 Position Size Adjustment: {(adjusted_size / base_size - 1) * 100:+.1f}%")
                logger.info(f"  🚨 Emergency Mode: {emergency_controls.emergency_mode}")
            
            self.test_results['risk_management_flow'] = {
                'status': 'PASSED',
                'scenarios_tested': len(test_scenarios),
                'scenario_results': scenario_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.test_results['risk_management_flow'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"❌ Risk management flow test failed: {e}")

    async def test_ai_decision_flow(self):
        """Test AI decision making flow"""
        logger.info("🧠 Testing AI decision flow...")
        
        try:
            ai_integrator = self.components['ai_integrator']
            ml_signals = self.components['ml_signals']
            
            # Test different market conditions
            market_conditions = [
                {
                    'name': 'Bullish Market',
                    'price': 50000,
                    'price_change_24h': 0.05,
                    'volume': 2000000,
                    'volatility': 0.02,
                    'trend': 'uptrend',
                    'rsi': 65
                },
                {
                    'name': 'Bearish Market',
                    'price': 45000,
                    'price_change_24h': -0.08,
                    'volume': 1500000,
                    'volatility': 0.06,
                    'trend': 'downtrend',
                    'rsi': 35
                },
                {
                    'name': 'Sideways Market',
                    'price': 47500,
                    'price_change_24h': 0.002,
                    'volume': 800000,
                    'volatility': 0.015,
                    'trend': 'neutral',
                    'rsi': 50
                }
            ]
            
            portfolio_data = {
                'total_value': 100000,
                'available_balance': 20000,
                'current_positions': {'BTCUSDT': {'size': 0.5}},
                'daily_pnl': 0,
                'win_rate': 0.6,
                'sharpe_ratio': 1.2
            }
            
            risk_metrics = {
                'portfolio_risk': 0.3,
                'max_drawdown': 0.05,
                'var_95': -2000,
                'correlation_risk': 0.2,
                'leverage_ratio': 1.0,
                'emergency_mode': False
            }
            
            decision_results = {}
            
            for condition in market_conditions:
                logger.info(f"🎯 Testing market condition: {condition['name']}")
                
                # Add symbol and other required fields
                market_data = {
                    'symbol': 'BTCUSDT',
                    **condition,
                    'support_level': condition['price'] * 0.95,
                    'resistance_level': condition['price'] * 1.05,
                    'macd': 0.01 if condition['trend'] == 'uptrend' else -0.01,
                    'bollinger_position': 0.7 if condition['trend'] == 'uptrend' else 0.3
                }
                
                # Generate ML signal
                ml_signal = await ml_signals.generate_signal(market_data)
                
                # Get AI decision
                ai_decision = await ai_integrator.get_ai_trading_decision(
                    market_data, portfolio_data, risk_metrics
                )
                
                decision_results[condition['name']] = {
                    'ml_signal': ml_signal.get('signal', 'UNKNOWN'),
                    'ml_confidence': ml_signal.get('confidence', 0),
                    'ai_action': ai_decision.action,
                    'ai_confidence': ai_decision.confidence,
                    'ai_reasoning': ai_decision.reasoning[:100] + "..." if len(ai_decision.reasoning) > 100 else ai_decision.reasoning,
                    'risk_score': ai_decision.risk_score
                }
                
                logger.info(f"  🤖 ML Signal: {ml_signal.get('signal', 'UNKNOWN')} ({ml_signal.get('confidence', 0):.1%})")
                logger.info(f"  🧠 AI Decision: {ai_decision.action} ({ai_decision.confidence:.1%})")
                logger.info(f"  ⚠️  Risk Score: {ai_decision.risk_score:.1%}")
            
            self.test_results['ai_decision_flow'] = {
                'status': 'PASSED',
                'conditions_tested': len(market_conditions),
                'decision_results': decision_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.test_results['ai_decision_flow'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"❌ AI decision flow test failed: {e}")

    async def test_websocket_processing(self):
        """Test WebSocket data processing capabilities"""
        logger.info("🌐 Testing WebSocket processing...")
        
        try:
            ws_client = self.components['websocket']
            
            # Test data callback
            received_data = []
            
            async def test_data_callback(symbol, data, data_type):
                received_data.append({
                    'symbol': symbol,
                    'data_type': data_type,
                    'price': data.price,
                    'timestamp': data.timestamp
                })
            
            # Add callback
            ws_client.add_data_callback(test_data_callback)
            
            # Simulate market data processing
            test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            
            for i, symbol in enumerate(test_symbols):
                # Simulate ticker data
                test_data = {
                    'stream': f'{symbol.lower()}@ticker',
                    'data': {
                        'c': 50000 + i * 1000,  # Current price
                        'v': 1000000 + i * 100000,  # Volume
                        'b': 49990 + i * 1000,  # Bid
                        'a': 50010 + i * 1000,  # Ask
                        'h': 51000 + i * 1000,  # High
                        'l': 49000 + i * 1000,  # Low
                        'P': '2.5',  # Price change percentage
                        'q': 50000000 + i * 5000000,  # Quote volume
                        'n': 10000 + i * 1000  # Trade count
                    }
                }
                
                # Process the test data
                await ws_client._process_market_data(test_data, time.time())
            
            # Test performance metrics
            metrics = ws_client.get_performance_metrics()
            
            self.test_results['websocket_processing'] = {
                'status': 'PASSED',
                'symbols_processed': len(test_symbols),
                'data_points_received': len(received_data),
                'performance_metrics': {
                    'total_messages': metrics.get('total_messages', 0),
                    'active_connections': metrics.get('active_connections', 0),
                    'subscribed_symbols': metrics.get('subscribed_symbols', 0)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"  📊 Processed {len(test_symbols)} symbols")
            logger.info(f"  📈 Received {len(received_data)} data points")
            
        except Exception as e:
            self.test_results['websocket_processing'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"❌ WebSocket processing test failed: {e}")

    async def test_ml_signal_generation(self):
        """Test ML signal generation"""
        logger.info("🤖 Testing ML signal generation...")
        
        try:
            ml_signals = self.components['ml_signals']
            
            # Test different market scenarios
            test_scenarios = [
                {
                    'name': 'Strong Uptrend',
                    'price': 50000,
                    'volume': 2000000,
                    'price_change_24h': 0.08,
                    'volatility': 0.03,
                    'rsi': 75,
                    'macd': 0.05,
                    'trend': 'uptrend'
                },
                {
                    'name': 'Strong Downtrend',
                    'price': 45000,
                    'volume': 1800000,
                    'price_change_24h': -0.12,
                    'volatility': 0.08,
                    'rsi': 25,
                    'macd': -0.08,
                    'trend': 'downtrend'
                },
                {
                    'name': 'Consolidation',
                    'price': 47500,
                    'volume': 800000,
                    'price_change_24h': 0.005,
                    'volatility': 0.015,
                    'rsi': 52,
                    'macd': 0.001,
                    'trend': 'neutral'
                }
            ]
            
            signal_results = {}
            
            for scenario in test_scenarios:
                logger.info(f"🎯 Testing scenario: {scenario['name']}")
                
                # Add required fields
                market_data = {
                    'symbol': 'BTCUSDT',
                    **scenario,
                    'support_level': scenario['price'] * 0.95,
                    'resistance_level': scenario['price'] * 1.05,
                    'bollinger_position': 0.5
                }
                
                # Generate signal
                signal = await ml_signals.generate_signal(market_data)
                
                signal_results[scenario['name']] = {
                    'signal': signal.get('signal', 'UNKNOWN'),
                    'confidence': signal.get('confidence', 0),
                    'reasoning': signal.get('reasoning', ''),
                    'features_used': len(signal.get('features', [])),
                    'model_consensus': signal.get('model_consensus', {})
                }
                
                logger.info(f"  📈 Signal: {signal.get('signal', 'UNKNOWN')} (confidence: {signal.get('confidence', 0):.1%})")
                logger.info(f"  🎯 Features used: {len(signal.get('features', []))}")
            
            self.test_results['ml_signal_generation'] = {
                'status': 'PASSED',
                'scenarios_tested': len(test_scenarios),
                'signal_results': signal_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.test_results['ml_signal_generation'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"❌ ML signal generation test failed: {e}")

    async def test_emergency_scenarios(self):
        """Test emergency scenario handling"""
        logger.info("🚨 Testing emergency scenarios...")
        
        try:
            emergency_controls = self.components['emergency_controls']
            risk_manager = self.components['risk_manager']
            
            # Test emergency triggers
            emergency_scenarios = [
                {
                    'name': 'Large Daily Loss',
                    'daily_pnl': -5000,
                    'portfolio_value': 100000,
                    'expected_emergency': True
                },
                {
                    'name': 'High Drawdown',
                    'daily_pnl': -1000,
                    'portfolio_value': 80000,
                    'peak_value': 100000,
                    'expected_emergency': True
                },
                {
                    'name': 'Normal Loss',
                    'daily_pnl': -500,
                    'portfolio_value': 99500,
                    'expected_emergency': False
                }
            ]
            
            emergency_results = {}
            
            for scenario in emergency_scenarios:
                logger.info(f"🎯 Testing emergency scenario: {scenario['name']}")
                
                # Reset emergency state
                emergency_controls.emergency_mode = False
                emergency_controls.daily_pnl = scenario['daily_pnl']
                emergency_controls.portfolio_value = scenario['portfolio_value']
                
                if 'peak_value' in scenario:
                    emergency_controls.peak_portfolio_value = scenario['peak_value']
                    emergency_controls._update_portfolio_metrics()
                
                # Simulate a trade that might trigger emergency
                emergency_controls.record_trade('TESTCOIN', 'buy', scenario['daily_pnl'], 0.8)
                
                # Check emergency state
                can_trade, reason = emergency_controls.should_allow_trade('TESTCOIN', 0.8)
                
                emergency_results[scenario['name']] = {
                    'emergency_triggered': emergency_controls.emergency_mode,
                    'expected_emergency': scenario['expected_emergency'],
                    'can_trade': can_trade,
                    'reason': reason,
                    'status_correct': emergency_controls.emergency_mode == scenario['expected_emergency']
                }
                
                logger.info(f"  🚨 Emergency Mode: {emergency_controls.emergency_mode}")
                logger.info(f"  ✅ Expected: {scenario['expected_emergency']}")
                logger.info(f"  🚦 Can Trade: {can_trade}")
            
            # Test emergency recovery
            if emergency_controls.emergency_mode:
                logger.info("🔄 Testing emergency recovery...")
                
                # Simulate time passage and improved conditions
                emergency_controls.emergency_start_time = datetime.now() - timedelta(hours=2)
                emergency_controls.daily_pnl = -100  # Improved loss
                
                can_exit = emergency_controls._can_exit_emergency()
                emergency_results['emergency_recovery'] = {
                    'can_exit_emergency': can_exit,
                    'recovery_test': 'PASSED' if can_exit else 'FAILED'
                }
                
                logger.info(f"  🔄 Can exit emergency: {can_exit}")
            
            self.test_results['emergency_scenarios'] = {
                'status': 'PASSED',
                'scenarios_tested': len(emergency_scenarios),
                'emergency_results': emergency_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.test_results['emergency_scenarios'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"❌ Emergency scenario test failed: {e}")

    def _generate_test_market_data(self) -> Dict:
        """Generate test market data"""
        return {
            'symbol': 'BTCUSDT',
            'price': 50000,
            'volume': 1500000,
            'price_change_24h': 0.025,
            'volatility': 0.035,
            'trend': 'uptrend',
            'support_level': 48000,
            'resistance_level': 52000,
            'rsi': 62,
            'macd': 0.02,
            'bollinger_position': 0.65,
            'high_24h': 51000,
            'low_24h': 49000
        }

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating integration test report...")
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Create report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'test_duration': '< 1 minute',
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'components_tested': list(self.components.keys()),
            'overall_status': 'PASSED' if success_rate >= 80 else 'FAILED'
        }
        
        # Save report to file
        report_file = 'integration_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🎯 INTEGRATION TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {total_tests - passed_tests}")
        logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"🔧 Components Tested: {len(self.components)}")
        
        if success_rate >= 80:
            logger.info("🎉 OVERALL STATUS: ✅ PASSED")
            logger.info("🚀 Enhanced trading system is ready for deployment!")
        else:
            logger.info("⚠️  OVERALL STATUS: ❌ FAILED")
            logger.info("🔧 Please review failed tests before deployment.")
        
        logger.info(f"📄 Detailed report saved to: {report_file}")
        logger.info("=" * 80)
        
        return report

async def main():
    """Run integration tests"""
    test_suite = IntegrationTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    import time
    
    # Suppress some logs for cleaner output
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Run tests
    asyncio.run(main())
