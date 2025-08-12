#!/usr/bin/env python3
"""
Enhanced Stop Loss System - Advanced risk management with multiple stop loss types.

Features:
- Dynamic ATR-based stops
- Trailing stops with profit protection
- Time-based stops for stuck positions
- PnL-based emergency stops
- Position size-based stops
- Market condition adaptive stops

Usage:
  python enhanced_stop_loss.py               # Single run
  python enhanced_stop_loss.py --monitor     # Continuous monitoring
  python enhanced_stop_loss.py --dry-run     # Test without real orders
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import pandas as pd
import numpy as np

# Ensure project modules importable
sys.path.append(os.path.dirname(__file__))

try:
    from trading_bot.config.settings import settings
    from trading_bot.utils import order_execution as oe
    from trading_bot.execution.position_manager import Position, PositionManager
    from trading_bot.exchange.binance_futures_gateway import BinanceAPIError
    from trading_bot.utils.binance_client import BinanceClient
except Exception as e:
    print(f"[FATAL] Import failed: {e}")
    sys.exit(1)

@dataclass
class StopLossConfig:
    """Enhanced stop loss configuration"""
    # Basic stop loss settings
    atr_multiplier: float = 2.0           # ATR-based stop distance
    max_loss_per_trade: float = 0.02      # 2% max loss per trade
    trailing_stop_enabled: bool = True    # Enable trailing stops
    trailing_distance: float = 0.015      # 1.5% trailing distance
    
    # Advanced stop loss settings
    time_based_stop_hours: int = 24       # Close position after 24 hours
    stuck_position_hours: int = 12        # Consider position stuck after 12 hours
    emergency_loss_threshold: float = 0.05  # 5% emergency stop
    profit_protection_threshold: float = 0.03  # Protect profits above 3%
    
    # Market condition stops
    volatility_stop_multiplier: float = 1.5  # Tighter stops in high vol
    volume_based_stops: bool = True       # Adjust stops based on volume
    correlation_stops: bool = True        # Portfolio correlation protection
    
    # Position management
    max_concurrent_positions: int = 8     # Maximum positions to hold
    position_size_scaling: bool = True    # Scale down position sizes
    drawdown_protection: bool = True      # Reduce position sizes on drawdown

@dataclass 
class StopLossSignal:
    """Stop loss signal information"""
    symbol: str
    action: str                          # 'close', 'reduce', 'trail'
    reason: str                          # Why stop was triggered
    stop_price: float                    # Calculated stop price
    urgency: str                         # 'low', 'medium', 'high', 'emergency'
    position_size: float                 # Current position size
    unrealized_pnl: float               # Current unrealized PnL
    recommended_action: str              # Specific action to take

class EnhancedStopLoss:
    """Enhanced stop loss system with multiple protection mechanisms"""
    
    def __init__(self, config: Optional[StopLossConfig] = None, dry_run: bool = False):
        self.config = config or StopLossConfig()
        self.dry_run = dry_run
        self.binance_client = BinanceClient()
        self.position_manager: Optional[PositionManager] = None
        self.active_symbols = set()
        self.price_cache = {}
        self.volatility_cache = {}
        self.last_portfolio_check = 0
        
        # Performance tracking
        self.stops_triggered = []
        self.performance_metrics = {
            'total_stops': 0,
            'profitable_stops': 0,
            'loss_prevented': 0.0,
            'avg_stop_time': 0.0
        }
        
        logger.info("🛡️ Enhanced Stop Loss System initialized")
        if dry_run:
            logger.warning("🧪 DRY RUN MODE - No real orders will be executed")
    
    def initialize(self):
        """Initialize the stop loss system"""
        try:
            # Initialize order execution
            oe.initialize()
            self.position_manager = oe._position_manager
            
            if not getattr(settings, 'FUTURES_ENABLED', False):
                logger.error("Futures trading not enabled")
                return False
                
            if not self.position_manager:
                logger.error("Position manager not available")
                return False
            
            # Load active symbols
            symbols_str = getattr(settings, 'FUTURES_SYMBOLS', '')
            self.active_symbols = {s.strip().upper() for s in symbols_str.split(',') if s.strip()}
            
            logger.info(f"📊 Monitoring {len(self.active_symbols)} symbols")
            logger.info(f"🎯 Active symbols: {sorted(list(self.active_symbols))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize stop loss system: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Tuple[float, float, float]:
        """Get current price, ATR, and volatility for symbol"""
        try:
            # Get current price
            ticker = self.binance_client.get_ticker_price(symbol)
            current_price = float(ticker['price'])
            
            # Get recent klines for ATR calculation
            klines = self.binance_client.get_klines(symbol, interval='1h', limit=30)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate ATR
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            atr = df['tr'].rolling(14).mean().iloc[-1]
            
            # Calculate volatility (20-period std)
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Cache the data
            self.price_cache[symbol] = current_price
            self.volatility_cache[symbol] = volatility
            
            return current_price, atr, volatility
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            # Return cached data if available
            price = self.price_cache.get(symbol, 0.0)
            return price, price * 0.01, 0.02  # Fallback values
    
    def calculate_atr_stop(self, position: Position, current_price: float, atr: float) -> float:
        """Calculate ATR-based stop loss price"""
        multiplier = self.config.atr_multiplier
        
        # Adjust multiplier based on volatility
        volatility = self.volatility_cache.get(position.symbol, 0.02)
        if volatility > 0.03:  # High volatility
            multiplier *= self.config.volatility_stop_multiplier
        
        if position.size > 0:  # Long position
            return position.entry_price - (atr * multiplier)
        else:  # Short position
            return position.entry_price + (atr * multiplier)
    
    def calculate_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """Calculate trailing stop price"""
        if not self.config.trailing_stop_enabled:
            return None
        
        if abs(position.size) < 1e-8:
            return None
        
        unrealized_pnl = position.unrealized_pnl(current_price)
        
        # Only trail if in profit above protection threshold
        if position.size > 0:  # Long position
            profit_pct = unrealized_pnl / (position.entry_price * position.size)
            if profit_pct > self.config.profit_protection_threshold:
                return current_price * (1 - self.config.trailing_distance)
        else:  # Short position
            profit_pct = -unrealized_pnl / (position.entry_price * abs(position.size))
            if profit_pct > self.config.profit_protection_threshold:
                return current_price * (1 + self.config.trailing_distance)
        
        return None
    
    def check_time_based_stops(self, position: Position) -> Optional[StopLossSignal]:
        """Check if position should be closed due to time"""
        position_age_hours = (time.time() - position.last_update) / 3600
        
        if position_age_hours > self.config.time_based_stop_hours:
            return StopLossSignal(
                symbol=position.symbol,
                action='close',
                reason=f'Time-based stop: position open for {position_age_hours:.1f} hours',
                stop_price=0.0,  # Market order
                urgency='medium',
                position_size=position.size,
                unrealized_pnl=0.0,
                recommended_action='market_close'
            )
        
        if position_age_hours > self.config.stuck_position_hours:
            current_price = self.price_cache.get(position.symbol, position.entry_price)
            unrealized_pnl = position.unrealized_pnl(current_price)
            
            # If stuck and losing, close it
            if unrealized_pnl < 0:
                return StopLossSignal(
                    symbol=position.symbol,
                    action='close',
                    reason=f'Stuck position: {position_age_hours:.1f}h old and losing',
                    stop_price=0.0,
                    urgency='high',
                    position_size=position.size,
                    unrealized_pnl=unrealized_pnl,
                    recommended_action='market_close'
                )
        
        return None
    
    def check_emergency_stops(self, position: Position, current_price: float) -> Optional[StopLossSignal]:
        """Check for emergency stop conditions"""
        unrealized_pnl = position.unrealized_pnl(current_price)
        position_value = abs(position.size * position.entry_price)
        
        if position_value > 0:
            loss_pct = -unrealized_pnl / position_value if unrealized_pnl < 0 else 0
            
            if loss_pct > self.config.emergency_loss_threshold:
                return StopLossSignal(
                    symbol=position.symbol,
                    action='close',
                    reason=f'EMERGENCY: Loss {loss_pct:.2%} exceeds threshold {self.config.emergency_loss_threshold:.2%}',
                    stop_price=0.0,
                    urgency='emergency',
                    position_size=position.size,
                    unrealized_pnl=unrealized_pnl,
                    recommended_action='immediate_market_close'
                )
        
        return None
    
    def check_portfolio_correlation(self) -> List[StopLossSignal]:
        """Check for portfolio-wide risk and correlation"""
        signals = []
        
        if not self.config.correlation_stops:
            return signals
        
        # Check if we need to reduce overall exposure
        total_positions = sum(1 for pos in self.position_manager.positions.values() 
                            if abs(pos.size) > 1e-8)
        
        if total_positions > self.config.max_concurrent_positions:
            # Find the worst performing positions to close
            positions_with_pnl = []
            for pos in self.position_manager.positions.values():
                if abs(pos.size) > 1e-8:
                    current_price = self.price_cache.get(pos.symbol, pos.entry_price)
                    unrealized_pnl = pos.unrealized_pnl(current_price)
                    positions_with_pnl.append((pos, unrealized_pnl))
            
            # Sort by PnL (worst first)
            positions_with_pnl.sort(key=lambda x: x[1])
            
            # Close worst performing positions
            excess_positions = total_positions - self.config.max_concurrent_positions
            for i in range(min(excess_positions, 3)):  # Close up to 3 at once
                pos, pnl = positions_with_pnl[i]
                signals.append(StopLossSignal(
                    symbol=pos.symbol,
                    action='close',
                    reason=f'Portfolio correlation: too many positions ({total_positions}), closing worst performer',
                    stop_price=0.0,
                    urgency='medium',
                    position_size=pos.size,
                    unrealized_pnl=pnl,
                    recommended_action='market_close'
                ))
        
        return signals
    
    def analyze_position(self, position: Position) -> List[StopLossSignal]:
        """Analyze a single position for stop loss conditions"""
        signals = []
        
        if abs(position.size) < 1e-8:
            return signals
        
        # Skip if symbol not in active list
        if position.symbol not in self.active_symbols:
            logger.debug(f"Skipping {position.symbol} - not in active symbols")
            return signals
        
        try:
            # Get market data
            current_price, atr, volatility = self.get_market_data(position.symbol)
            unrealized_pnl = position.unrealized_pnl(current_price)
            
            logger.debug(f"Analyzing {position.symbol}: size={position.size:.4f}, "
                        f"entry={position.entry_price:.4f}, current={current_price:.4f}, "
                        f"pnl={unrealized_pnl:.4f}")
            
            # 1. Emergency stops (highest priority)
            emergency_signal = self.check_emergency_stops(position, current_price)
            if emergency_signal:
                signals.append(emergency_signal)
                return signals  # Emergency stops override everything
            
            # 2. Time-based stops
            time_signal = self.check_time_based_stops(position)
            if time_signal:
                signals.append(time_signal)
            
            # 3. ATR-based stops
            atr_stop_price = self.calculate_atr_stop(position, current_price, atr)
            stop_triggered = False
            
            if position.size > 0:  # Long position
                stop_triggered = current_price <= atr_stop_price
            else:  # Short position
                stop_triggered = current_price >= atr_stop_price
            
            if stop_triggered:
                signals.append(StopLossSignal(
                    symbol=position.symbol,
                    action='close',
                    reason=f'ATR stop triggered: price {current_price:.4f} vs stop {atr_stop_price:.4f}',
                    stop_price=atr_stop_price,
                    urgency='high',
                    position_size=position.size,
                    unrealized_pnl=unrealized_pnl,
                    recommended_action='limit_close'
                ))
            
            # 4. Trailing stops
            trailing_stop_price = self.calculate_trailing_stop(position, current_price)
            if trailing_stop_price:
                trail_triggered = False
                if position.size > 0:  # Long position
                    trail_triggered = current_price <= trailing_stop_price
                else:  # Short position
                    trail_triggered = current_price >= trailing_stop_price
                
                if trail_triggered:
                    signals.append(StopLossSignal(
                        symbol=position.symbol,
                        action='close',
                        reason=f'Trailing stop triggered: price {current_price:.4f} vs trail {trailing_stop_price:.4f}',
                        stop_price=trailing_stop_price,
                        urgency='medium',
                        position_size=position.size,
                        unrealized_pnl=unrealized_pnl,
                        recommended_action='limit_close'
                    ))
            
        except Exception as e:
            logger.error(f"Error analyzing position {position.symbol}: {e}")
        
        return signals
    
    def execute_stop_loss(self, signal: StopLossSignal) -> bool:
        """Execute a stop loss signal"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute {signal.action} for {signal.symbol}: {signal.reason}")
            return True
        
        try:
            position = self.position_manager.get_position(signal.symbol)
            
            if abs(position.size) < 1e-8:
                logger.warning(f"No position to close for {signal.symbol}")
                return False
            
            # Determine order side and quantity
            side = 'SELL' if position.size > 0 else 'BUY'
            qty = abs(position.size)
            
            logger.warning(f"🛑 STOP LOSS TRIGGERED: {signal.symbol}")
            logger.warning(f"   Reason: {signal.reason}")
            logger.warning(f"   Urgency: {signal.urgency}")
            logger.warning(f"   Action: {side} {qty:.8f} {signal.symbol}")
            
            # Execute the order
            if oe._gateway:
                order_type = 'MARKET' if signal.urgency == 'emergency' else 'LIMIT'
                price = None if order_type == 'MARKET' else signal.stop_price
                
                order = oe._gateway.create_order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=qty,
                    order_type=order_type,
                    price=price,
                    reduce_only=True
                )
                
                # Update position
                fill_qty = qty if side == 'BUY' else -qty
                fill_price = float(order.get('avgPrice', position.entry_price or 0) or 0)
                position.update_fill(fill_qty, fill_price)
                
                # Track performance
                self.performance_metrics['total_stops'] += 1
                if signal.unrealized_pnl > 0:
                    self.performance_metrics['profitable_stops'] += 1
                self.performance_metrics['loss_prevented'] += max(0, -signal.unrealized_pnl)
                
                # Record the stop
                self.stops_triggered.append({
                    'timestamp': datetime.now(),
                    'symbol': signal.symbol,
                    'reason': signal.reason,
                    'size': position.size,
                    'pnl': signal.unrealized_pnl,
                    'order_id': order.get('orderId')
                })
                
                logger.success(f"✅ Stop loss executed: {signal.symbol} order={order.get('orderId')}")
                return True
            else:
                logger.error("Gateway not available for stop loss execution")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute stop loss for {signal.symbol}: {e}")
            return False
    
    def run_stop_loss_check(self) -> Dict:
        """Run a complete stop loss check on all positions"""
        start_time = time.time()
        results = {
            'positions_checked': 0,
            'signals_generated': 0,
            'stops_executed': 0,
            'errors': 0
        }
        
        if not self.position_manager:
            logger.error("Position manager not available")
            return results
        
        logger.info("🔍 Running stop loss analysis...")
        
        all_signals = []
        
        # Check individual positions
        for symbol, position in self.position_manager.positions.items():
            if abs(position.size) > 1e-8:
                results['positions_checked'] += 1
                try:
                    signals = self.analyze_position(position)
                    all_signals.extend(signals)
                    results['signals_generated'] += len(signals)
                except Exception as e:
                    logger.error(f"Error checking position {symbol}: {e}")
                    results['errors'] += 1
        
        # Check portfolio-level stops
        portfolio_signals = self.check_portfolio_correlation()
        all_signals.extend(portfolio_signals)
        results['signals_generated'] += len(portfolio_signals)
        
        # Sort signals by urgency
        urgency_order = {'emergency': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_signals.sort(key=lambda s: urgency_order.get(s.urgency, 3))
        
        # Execute stop losses
        for signal in all_signals:
            if self.execute_stop_loss(signal):
                results['stops_executed'] += 1
            time.sleep(0.5)  # Brief pause between executions
        
        execution_time = time.time() - start_time
        
        logger.info(f"📊 Stop loss check complete:")
        logger.info(f"   ⏱️ Execution time: {execution_time:.2f}s")
        logger.info(f"   📋 Positions checked: {results['positions_checked']}")
        logger.info(f"   🚨 Signals generated: {results['signals_generated']}")
        logger.info(f"   🛑 Stops executed: {results['stops_executed']}")
        if results['errors'] > 0:
            logger.warning(f"   ❌ Errors: {results['errors']}")
        
        return results
    
    def print_performance_summary(self):
        """Print performance summary"""
        metrics = self.performance_metrics
        total_stops = metrics['total_stops']
        
        if total_stops == 0:
            logger.info("📈 No stops triggered yet")
            return
        
        profitable_pct = (metrics['profitable_stops'] / total_stops) * 100
        
        logger.info("📊 STOP LOSS PERFORMANCE SUMMARY")
        logger.info("="*50)
        logger.info(f"🛑 Total stops triggered: {total_stops}")
        logger.info(f"💰 Profitable stops: {metrics['profitable_stops']} ({profitable_pct:.1f}%)")
        logger.info(f"🛡️ Total loss prevented: ${metrics['loss_prevented']:.2f}")
        logger.info(f"⏱️ Average stop time: {metrics['avg_stop_time']:.1f}h")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Stop Loss System')
    parser.add_argument('--dry-run', action='store_true', help='Test mode without real orders')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--config', help='Path to stop loss config file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("enhanced_stop_loss.log", rotation="10 MB", retention="7 days")
    
    # Initialize enhanced stop loss
    config = StopLossConfig()
    stop_loss = EnhancedStopLoss(config=config, dry_run=args.dry_run)
    
    if not stop_loss.initialize():
        logger.error("Failed to initialize stop loss system")
        return 1
    
    logger.info("🚀 Enhanced Stop Loss System started")
    
    try:
        if args.monitor:
            logger.info(f"📡 Monitoring mode: checking every {args.interval} seconds")
            while True:
                try:
                    stop_loss.run_stop_loss_check()
                    time.sleep(args.interval)
                except KeyboardInterrupt:
                    logger.info("🛑 Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(30)  # Wait before retrying
        else:
            # Single run
            results = stop_loss.run_stop_loss_check()
            stop_loss.print_performance_summary()
            
            if results['stops_executed'] > 0:
                logger.warning(f"⚠️ {results['stops_executed']} stop losses were executed!")
            else:
                logger.info("✅ No stop losses triggered - all positions within risk limits")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
