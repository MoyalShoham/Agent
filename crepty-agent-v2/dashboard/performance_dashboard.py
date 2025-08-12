"""
Performance Dashboard - Real-time monitoring of trading performance and risk metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_trade_data():
    """Load trade log data"""
    trade_files = ['trade_log.csv', 'futures_trades_log.csv', 'trade_log_clean.csv']
    
    for file in trade_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
    
    return pd.DataFrame()

def calculate_performance_metrics(trades_df):
    """Calculate comprehensive performance metrics"""
    if trades_df.empty:
        return {}
    
    # Basic metrics
    total_trades = len(trades_df)
    
    # PnL calculations
    if 'realized_pnl' in trades_df.columns:
        pnl_data = trades_df['realized_pnl'].dropna()
        if len(pnl_data) > 0:
            total_pnl = pnl_data.sum()
            winning_trades = (pnl_data > 0).sum()
            losing_trades = (pnl_data < 0).sum()
            win_rate = winning_trades / len(pnl_data) * 100 if len(pnl_data) > 0 else 0
            
            avg_win = pnl_data[pnl_data > 0].mean() if winning_trades > 0 else 0
            avg_loss = pnl_data[pnl_data < 0].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / avg_loss / losing_trades) if avg_loss != 0 and losing_trades > 0 else 0
        else:
            total_pnl = win_rate = profit_factor = avg_win = avg_loss = 0
    else:
        total_pnl = win_rate = profit_factor = avg_win = avg_loss = 0
    
    # Portfolio value calculation
    if 'current_total_usdt' in trades_df.columns:
        portfolio_values = trades_df['current_total_usdt'].dropna()
        if len(portfolio_values) > 1:
            returns = portfolio_values.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized %
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Maximum drawdown
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak * 100
            max_drawdown = drawdown.min()
        else:
            volatility = sharpe_ratio = max_drawdown = 0
    else:
        volatility = sharpe_ratio = max_drawdown = 0
    
    return {
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def create_pnl_chart(trades_df):
    """Create PnL over time chart"""
    if trades_df.empty or 'timestamp' not in trades_df.columns:
        return go.Figure()
    
    # Calculate cumulative PnL
    if 'realized_pnl' in trades_df.columns:
        trades_df = trades_df.sort_values('timestamp')
        cumulative_pnl = trades_df['realized_pnl'].fillna(0).cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Cumulative PnL Over Time',
            xaxis_title='Date',
            yaxis_title='PnL (USDT)',
            template='plotly_dark'
        )
        
        return fig
    
    return go.Figure()

def create_portfolio_value_chart(trades_df):
    """Create portfolio value over time chart"""
    if trades_df.empty or 'current_total_usdt' not in trades_df.columns:
        return go.Figure()
    
    trades_df = trades_df.sort_values('timestamp')
    portfolio_values = trades_df[['timestamp', 'current_total_usdt']].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_values['timestamp'],
        y=portfolio_values['current_total_usdt'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Value (USDT)',
        template='plotly_dark'
    )
    
    return fig

def create_symbol_performance_chart(trades_df):
    """Create symbol performance breakdown"""
    if trades_df.empty or 'symbol' not in trades_df.columns:
        return go.Figure()
    
    if 'realized_pnl' in trades_df.columns:
        symbol_pnl = trades_df.groupby('symbol')['realized_pnl'].sum().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=symbol_pnl.index, y=symbol_pnl.values,
                  marker_color=['green' if x > 0 else 'red' for x in symbol_pnl.values])
        ])
        
        fig.update_layout(
            title='PnL by Symbol',
            xaxis_title='Symbol',
            yaxis_title='Total PnL (USDT)',
            template='plotly_dark'
        )
        
        return fig
    
    return go.Figure()

def create_strategy_performance_chart(trades_df):
    """Create strategy performance breakdown"""
    if trades_df.empty or 'strategy' not in trades_df.columns:
        return go.Figure()
    
    if 'realized_pnl' in trades_df.columns:
        strategy_pnl = trades_df.groupby('strategy')['realized_pnl'].sum().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=strategy_pnl.index, y=strategy_pnl.values,
                  marker_color=['green' if x > 0 else 'red' for x in strategy_pnl.values])
        ])
        
        fig.update_layout(
            title='PnL by Strategy',
            xaxis_title='Strategy',
            yaxis_title='Total PnL (USDT)',
            template='plotly_dark'
        )
        
        return fig
    
    return go.Figure()

def main():
    st.set_page_config(
        page_title="Crypto Trading Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🚀 Crypto Trading Agent Dashboard")
    st.markdown("Real-time performance monitoring and risk analytics")
    
    # Load data
    trades_df = load_trade_data()
    
    if trades_df.empty:
        st.warning("No trade data found. Make sure your trading agent is running and generating trade logs.")
        st.info("Expected files: trade_log.csv, futures_trades_log.csv, or trade_log_clean.csv")
        return
    
    # Calculate metrics
    metrics = calculate_performance_metrics(trades_df)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics.get('total_trades', 0))
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    
    with col2:
        pnl = metrics.get('total_pnl', 0)
        st.metric("Total PnL", f"${pnl:,.2f}", 
                 delta=None, delta_color="normal")
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    
    with col3:
        st.metric("Avg Win", f"${metrics.get('avg_win', 0):.2f}")
        st.metric("Avg Loss", f"${metrics.get('avg_loss', 0):.2f}")
    
    with col4:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1f}%")
    
    # Charts
    st.markdown("## 📊 Performance Charts")
    
    # PnL Chart
    col1, col2 = st.columns(2)
    
    with col1:
        pnl_chart = create_pnl_chart(trades_df)
        st.plotly_chart(pnl_chart, use_container_width=True)
    
    with col2:
        portfolio_chart = create_portfolio_value_chart(trades_df)
        st.plotly_chart(portfolio_chart, use_container_width=True)
    
    # Performance Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_chart = create_symbol_performance_chart(trades_df)
        st.plotly_chart(symbol_chart, use_container_width=True)
    
    with col2:
        strategy_chart = create_strategy_performance_chart(trades_df)
        st.plotly_chart(strategy_chart, use_container_width=True)
    
    # Recent trades table
    st.markdown("## 📋 Recent Trades")
    
    if not trades_df.empty:
        # Show last 20 trades
        recent_trades = trades_df.sort_values('timestamp', ascending=False).head(20)
        display_cols = ['timestamp', 'action', 'symbol', 'qty', 'price', 'status', 'strategy']
        available_cols = [col for col in display_cols if col in recent_trades.columns]
        
        if available_cols:
            st.dataframe(recent_trades[available_cols], use_container_width=True)
    
    # Risk metrics (if available)
    try:
        from trading_bot.risk.enhanced_risk_manager import EnhancedRiskManager
        
        st.markdown("## ⚠️ Risk Metrics")
        
        # This would be populated by the actual risk manager in a live system
        st.info("Connect to live risk manager for real-time risk metrics")
        
    except ImportError:
        pass
    
    # Auto-refresh
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
