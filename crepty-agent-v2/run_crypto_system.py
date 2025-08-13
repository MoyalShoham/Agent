#!/usr/bin/env python3
"""
Comprehensive Crypto Trading System Launcher
Auto-implements all AI agents, data collection, and trading system
"""

import asyncio
import os
import sys
import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "trading_bot"))

from loguru import logger
from trading_bot.ai_models.ai_agents import create_ai_agent_orchestrator
from trading_bot.data.real_time_data_collector import RealTimeDataManager
from trading_bot.strategies.strategy_experts import StrategyManager
from trading_bot.ai_models.meta_learner import MetaLearner
from trading_bot.coordinators.trading_system_coordinator import TradingSystemCoordinator

class CryptoSystemLauncher:
    """Automated launcher for the complete crypto trading system"""
    
    def __init__(self):
        self.setup_logging()
        self.config = self.load_configuration()
        self.components = {}
        self.running = False
        
        # Initialize system components
        self.initialize_components()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console logging
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # Add file logging
        logger.add(
            log_dir / "crypto_system_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        # Add error file logging
        logger.add(
            log_dir / "crypto_system_errors_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR"
        )
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        config_file = Path("config.json")
        
        default_config = {
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
                "position_size": 0.1,
                "max_positions": 5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            },
            "data_collection": {
                "enabled": True,
                "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "websocket_streams": True
            },
            "ai_agents": {
                "enabled": True,
                "analysis_interval": 300,  # 5 minutes
                "comprehensive_analysis_interval": 900  # 15 minutes
            },
            "reports": {
                "monthly_auto_generate": True,
                "yearly_auto_generate": True,
                "monthly_day": 1,  # First day of month
                "email_reports": False
            },
            "risk_management": {
                "max_portfolio_risk": 0.05,
                "max_single_position_risk": 0.02,
                "emergency_stop_loss": 0.10
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in loaded_config:
                            loaded_config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in loaded_config[key]:
                                    loaded_config[key][subkey] = subvalue
                    return loaded_config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        logger.info("Created default configuration file")
        return default_config
        
    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Crypto Trading System Components...")
            
            # Get symbols from environment or use defaults
            symbols_str = os.getenv('FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOTUSDT,NEARUSDT')
            symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
            
            # Initialize Trading System Coordinator (it will initialize all other components)
            logger.info("⚡ Initializing Trading System Coordinator...")
            self.components['coordinator'] = TradingSystemCoordinator(
                symbols=symbols,
                config=self.config
            )
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
            
    async def start_system(self):
        """Start the complete crypto trading system"""
        try:
            logger.info("🚀 Starting Comprehensive Crypto Trading System...")
            
            # Initialize the coordinator (which initializes all components)
            await self.components['coordinator'].initialize()
            
            # Start trading system
            logger.info("⚡ Starting trading system coordinator...")
            trading_task = asyncio.create_task(
                self.components['coordinator'].run()
            )
            
            # Schedule AI analysis
            self.schedule_ai_analysis()
            
            # Schedule automatic reports
            self.schedule_automatic_reports()
            
            # Start scheduler in background
            scheduler_task = asyncio.create_task(self.run_scheduler())
            
            self.running = True
            logger.info("✅ Crypto Trading System is now running!")
            
            # Wait for tasks
            await asyncio.gather(trading_task, scheduler_task)
            
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            raise
            
    def schedule_ai_analysis(self):
        """Schedule regular AI analysis"""
        if not self.config['ai_agents']['enabled']:
            return
            
        # Quick analysis every 5 minutes
        schedule.every(self.config['ai_agents']['analysis_interval']).seconds.do(
            self.run_quick_ai_analysis
        )
        
        # Comprehensive analysis every 15 minutes
        schedule.every(self.config['ai_agents']['comprehensive_analysis_interval']).seconds.do(
            self.run_comprehensive_ai_analysis
        )
        
        logger.info("📊 AI analysis scheduled")
        
    def schedule_automatic_reports(self):
        """Schedule automatic report generation"""
        if not self.config['reports']['monthly_auto_generate']:
            return
            
        # Monthly reports on the first day of every month
        schedule.every().day.at("00:01").do(self.check_monthly_report)
        
        # Daily check for yearly reports (will generate on Jan 1st)
        schedule.every().day.at("00:05").do(self.check_yearly_report)
        
        logger.info("📋 Automatic reports scheduled")
        
    def check_monthly_report(self):
        """Check if monthly report should be generated"""
        today = datetime.now()
        if today.day == 1:  # First day of the month
            self.generate_monthly_report()
            
    def check_yearly_report(self):
        """Check if yearly report should be generated"""
        today = datetime.now()
        if today.month == 1 and today.day == 1:  # January 1st
            self.generate_yearly_report()
        
    async def run_scheduler(self):
        """Run the scheduled tasks"""
        while self.running:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
            
    def run_quick_ai_analysis(self):
        """Run quick AI analysis"""
        try:
            logger.info("📊 Running quick AI analysis...")
            asyncio.create_task(self._async_quick_analysis())
        except Exception as e:
            logger.error(f"Quick AI analysis error: {e}")
            
    async def _async_quick_analysis(self):
        """Async quick analysis"""
        try:
            # Get AI orchestrator from coordinator
            ai_orchestrator = self.components['coordinator'].ai_orchestrator
            if not ai_orchestrator:
                logger.warning("AI orchestrator not available")
                return
                
            # Get latest market data from coordinator
            market_data = await self.components['coordinator']._collect_market_data('BTCUSDT')
            
            if market_data:
                # Run quick analysis
                analysis = await ai_orchestrator.get_comprehensive_analysis(
                    "Quick market analysis", market_data
                )
                
                logger.info("✅ Quick AI analysis completed")
            else:
                logger.warning("No market data available for analysis")
            
        except Exception as e:
            logger.error(f"Async quick analysis error: {e}")
            
    def run_comprehensive_ai_analysis(self):
        """Run comprehensive AI analysis"""
        try:
            logger.info("📊 Running comprehensive AI analysis...")
            asyncio.create_task(self._async_comprehensive_analysis())
        except Exception as e:
            logger.error(f"Comprehensive AI analysis error: {e}")
            
    async def _async_comprehensive_analysis(self):
        """Async comprehensive analysis"""
        try:
            # Get AI orchestrator from coordinator
            ai_orchestrator = self.components['coordinator'].ai_orchestrator
            if not ai_orchestrator:
                logger.warning("AI orchestrator not available")
                return
                
            # Get latest market data from coordinator
            market_data = await self.components['coordinator']._collect_market_data('BTCUSDT')
            
            if market_data:
                # Run comprehensive analysis
                analysis = await ai_orchestrator.get_comprehensive_analysis(
                    "Comprehensive market analysis", market_data
                )
                
                # Get trading recommendations
                portfolio_data = {"total_value": 10000, "positions": []}
                trading_rec = await ai_orchestrator.get_trading_recommendation(
                    "Trading recommendation based on comprehensive analysis", 
                    {'market_data': market_data, 'portfolio_data': portfolio_data}
                )
                
                logger.info("✅ Comprehensive AI analysis completed")
            else:
                logger.warning("No market data available for analysis")
            
        except Exception as e:
            logger.error(f"Async comprehensive analysis error: {e}")
            
    def generate_monthly_report(self):
        """Generate monthly financial report"""
        try:
            logger.info("📋 Generating monthly financial report...")
            asyncio.create_task(self._async_monthly_report())
        except Exception as e:
            logger.error(f"Monthly report generation error: {e}")
            
    async def _async_monthly_report(self):
        """Async monthly report generation"""
        try:
            # Get AI orchestrator from coordinator
            ai_orchestrator = self.components['coordinator'].ai_orchestrator
            if not ai_orchestrator:
                logger.warning("AI orchestrator not available")
                return
                
            # Get trading data (mock for now)
            trading_data = {
                "trades": [],
                "total_pnl": 0,
                "period": "monthly"
            }
            
            # Generate monthly report
            report = await ai_orchestrator.generate_monthly_financial_report(
                "Generate comprehensive monthly financial report", trading_data
            )
            
            logger.info("✅ Monthly financial report generated")
            
        except Exception as e:
            logger.error(f"Async monthly report error: {e}")
            
    def generate_yearly_report(self):
        """Generate yearly financial report"""
        try:
            logger.info("📋 Generating yearly financial report...")
            asyncio.create_task(self._async_yearly_report())
        except Exception as e:
            logger.error(f"Yearly report generation error: {e}")
            
    async def _async_yearly_report(self):
        """Async yearly report generation"""
        try:
            # Get AI orchestrator from coordinator
            ai_orchestrator = self.components['coordinator'].ai_orchestrator
            if not ai_orchestrator:
                logger.warning("AI orchestrator not available")
                return
                
            # Get trading data (mock for now)
            trading_data = {
                "trades": [],
                "total_pnl": 0,
                "period": "yearly"
            }
            
            # Generate yearly report
            report = await ai_orchestrator.generate_yearly_financial_report(
                "Generate comprehensive yearly financial report", trading_data
            )
            
            # Also generate tax reports
            tax_reports = await ai_orchestrator.generate_tax_reports(trading_data)
            
            logger.info("✅ Yearly financial and tax reports generated")
            
        except Exception as e:
            logger.error(f"Async yearly report error: {e}")
            
    async def stop_system(self):
        """Stop the trading system gracefully"""
        try:
            logger.info("🛑 Stopping Crypto Trading System...")
            
            self.running = False
            
            # Stop coordinator (which will stop all components)
            if hasattr(self.components.get('coordinator'), 'shutdown'):
                await self.components['coordinator'].shutdown()
                
            logger.info("✅ System stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.running,
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'config': self.config
        }
        
        # Check component status
        for name, component in self.components.items():
            if hasattr(component, 'get_status'):
                status['components'][name] = component.get_status()
            else:
                status['components'][name] = {'initialized': True}
                
        return status
        
    def print_startup_info(self):
        """Print system startup information"""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE CRYPTO TRADING SYSTEM")
        print("="*80)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Symbols: {', '.join(self.config['trading']['symbols'])}")
        print(f"🤖 AI Agents: {'Enabled' if self.config['ai_agents']['enabled'] else 'Disabled'}")
        print(f"📡 Data Collection: {'Enabled' if self.config['data_collection']['enabled'] else 'Disabled'}")
        print(f"📋 Auto Reports: {'Enabled' if self.config['reports']['monthly_auto_generate'] else 'Disabled'}")
        print("="*80)
        print("System Components:")
        print("  📊 AI Agent Orchestrator (8 Expert Agents)")
        print("  📡 Real-time Data Collector (Binance WebSocket)")
        print("  🎯 Strategy Manager (4 Trading Strategies)")
        print("  🧠 Meta Learner (Contextual Bandits)")
        print("  ⚡ Trading System Coordinator")
        print("  📋 Automatic Report Generation")
        print("="*80)
        print("🔄 System is starting...")
        print("\n")

async def main():
    """Main entry point"""
    launcher = CryptoSystemLauncher()
    
    try:
        launcher.print_startup_info()
        await launcher.start_system()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
        await launcher.stop_system()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        await launcher.stop_system()
        sys.exit(1)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
        
    # Check for required environment variables
    required_env_vars = ["OPENAI_API_KEY", "BINANCE_API_KEY", "BINANCE_SECRET_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        sys.exit(1)
        
    # Run the system
    asyncio.run(main())
