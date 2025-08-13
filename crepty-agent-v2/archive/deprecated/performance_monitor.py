#!/usr/bin/env python3
"""
Performance Monitor - Simple command-line performance monitoring tool.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_trade_data():
    """Load trade log data from available files"""
    trade_files = [
        'trade_log.csv',
        'futures_trades_log.csv', 
        'trade_log_clean.csv',
        'trade_log_clean_fixed.csv'
    ]
    
    for file in trade_files:
        if os.path.exists(file):
            try:
                logger.info(f"Loading trade data from {file}")
                df = pd.read_csv(file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Loaded {len(df)} trades from {file}")
                return df, file
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
    
    return pd.DataFrame(), None

def calculate_basic_metrics(trades_df):
    """Calculate basic trading performance metrics"""
    if trades_df.empty:
        return {}
    
    metrics = {
        'total_trades': len(trades_df),
        'date_range': 'N/A',
        'symbols_traded': 0,
        'strategies_used': 0
    }
    
    # Date range
    if 'timestamp' in trades_df.columns:
        min_date = trades_df['timestamp'].min()
        max_date = trades_df['timestamp'].max()
        metrics['date_range'] = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    
    # Symbols
    if 'symbol' in trades_df.columns:
        metrics['symbols_traded'] = trades_df['symbol'].nunique()
        metrics['top_symbols'] = trades_df['symbol'].value_counts().head(5).to_dict()
    
    # Strategies
    if 'strategy' in trades_df.columns:
        metrics['strategies_used'] = trades_df['strategy'].nunique()
        metrics['strategy_distribution'] = trades_df['strategy'].value_counts().to_dict()
    
    # PnL analysis
    if 'realized_pnl' in trades_df.columns:
        pnl_data = pd.to_numeric(trades_df['realized_pnl'], errors='coerce').dropna()
        if len(pnl_data) > 0:
            metrics['total_pnl'] = pnl_data.sum()
            metrics['avg_pnl'] = pnl_data.mean()
            metrics['winning_trades'] = (pnl_data > 0).sum()
            metrics['losing_trades'] = (pnl_data < 0).sum()
            metrics['win_rate'] = (metrics['winning_trades'] / len(pnl_data) * 100) if len(pnl_data) > 0 else 0
            
            if metrics['winning_trades'] > 0:
                metrics['avg_win'] = pnl_data[pnl_data > 0].mean()
            if metrics['losing_trades'] > 0:
                metrics['avg_loss'] = pnl_data[pnl_data < 0].mean()
    
    # Portfolio value analysis
    if 'current_total_usdt' in trades_df.columns:
        portfolio_values = pd.to_numeric(trades_df['current_total_usdt'], errors='coerce').dropna()
        if len(portfolio_values) > 1:
            metrics['initial_portfolio'] = portfolio_values.iloc[0]
            metrics['final_portfolio'] = portfolio_values.iloc[-1]
            metrics['portfolio_return'] = ((metrics['final_portfolio'] / metrics['initial_portfolio']) - 1) * 100
            
            # Calculate simple volatility
            returns = portfolio_values.pct_change().dropna()
            if len(returns) > 0:
                metrics['portfolio_volatility'] = returns.std() * 100  # Daily volatility %
    
    return metrics

def print_performance_report(metrics, data_source):
    """Print a formatted performance report"""
    print("\n" + "="*60)
    print("📊 TRADING PERFORMANCE REPORT")
    print("="*60)
    
    if not metrics:
        print("❌ No trading data found or data could not be processed")
        return
    
    print(f"📁 Data Source: {data_source}")
    print(f"📅 Date Range: {metrics.get('date_range', 'N/A')}")
    print(f"📈 Total Trades: {metrics.get('total_trades', 0):,}")
    
    print("\n💰 PROFITABILITY")
    print("-" * 40)
    if 'total_pnl' in metrics:
        print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
        print(f"Average PnL: ${metrics.get('avg_pnl', 0):.2f}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        
        if 'avg_win' in metrics:
            print(f"Average Win: ${metrics['avg_win']:.2f}")
        if 'avg_loss' in metrics:
            print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    else:
        print("No PnL data available")
    
    print("\n💼 PORTFOLIO")
    print("-" * 40)
    if 'initial_portfolio' in metrics:
        print(f"Initial Value: ${metrics['initial_portfolio']:,.2f}")
        print(f"Final Value: ${metrics['final_portfolio']:,.2f}")
        print(f"Total Return: {metrics.get('portfolio_return', 0):.2f}%")
        print(f"Volatility: {metrics.get('portfolio_volatility', 0):.2f}%")
    else:
        print("No portfolio data available")
    
    print("\n🎯 TRADING ACTIVITY")
    print("-" * 40)
    print(f"Symbols Traded: {metrics.get('symbols_traded', 0)}")
    print(f"Strategies Used: {metrics.get('strategies_used', 0)}")
    
    # Top symbols
    if 'top_symbols' in metrics:
        print("\nTop 5 Symbols by Trade Count:")
        for symbol, count in list(metrics['top_symbols'].items())[:5]:
            print(f"  {symbol}: {count} trades")
    
    # Strategy distribution
    if 'strategy_distribution' in metrics:
        print("\nStrategy Distribution:")
        for strategy, count in metrics['strategy_distribution'].items():
            print(f"  {strategy}: {count} trades")
    
    print("\n" + "="*60)

def check_recent_activity(trades_df):
    """Check for recent trading activity"""
    if trades_df.empty or 'timestamp' not in trades_df.columns:
        return
    
    now = datetime.now()
    
    # Recent activity (last 24 hours)
    recent_cutoff = now - timedelta(hours=24)
    recent_trades = trades_df[trades_df['timestamp'] > recent_cutoff]
    
    print("\n🕐 RECENT ACTIVITY (Last 24 Hours)")
    print("-" * 40)
    print(f"Recent Trades: {len(recent_trades)}")
    
    if len(recent_trades) > 0:
        last_trade = trades_df.iloc[-1]
        time_since_last = now - last_trade['timestamp']
        
        print(f"Last Trade: {time_since_last.total_seconds() / 3600:.1f} hours ago")
        print(f"Last Symbol: {last_trade.get('symbol', 'N/A')}")
        print(f"Last Action: {last_trade.get('action', 'N/A')}")
        
        if 'realized_pnl' in recent_trades.columns:
            recent_pnl = pd.to_numeric(recent_trades['realized_pnl'], errors='coerce').sum()
            print(f"Recent PnL: ${recent_pnl:.2f}")
    else:
        print("No recent trading activity")

def check_risk_status():
    """Check current risk status"""
    print("\n⚠️ RISK STATUS")
    print("-" * 40)
    
    try:
        from trading_bot.risk.risk_manager import RiskManager
        
        # This would normally be connected to live risk manager
        print("Risk manager available but not connected to live data")
        print("Connect to live system for real-time risk metrics")
        
    except ImportError:
        print("Risk manager module not available")

def main():
    """Main monitoring function"""
    print("🔍 Loading trading performance data...")
    
    # Load trade data
    trades_df, data_source = load_trade_data()
    
    if trades_df.empty:
        print("❌ No trade data found!")
        print("\nExpected files:")
        print("- trade_log.csv")
        print("- futures_trades_log.csv")
        print("- trade_log_clean.csv")
        print("\nMake sure your trading agent is running and generating logs.")
        return
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_basic_metrics(trades_df)
    
    # Print report
    print_performance_report(metrics, data_source)
    
    # Check recent activity
    check_recent_activity(trades_df)
    
    # Check risk status
    check_risk_status()
    
    print(f"\n📊 Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Run this script anytime to get updated performance metrics!")

if __name__ == "__main__":
    main()
