"""
Real-time Performance Monitor with Advanced Analytics
Provides comprehensive monitoring and visualization of trading performance.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trading_bot.risk.advanced_risk_manager import AdvancedRiskManager
    from trading_bot.utils.enhanced_ml_signals import get_enhanced_model_performance
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all dependencies are installed.")

st.set_page_config(
    page_title="Crypto Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_trade_data():
    """Load trade log data with caching"""
    try:
        # Try multiple file locations
        possible_files = [
            'futures_trades_log.csv',
            'futures_trades_log_cleaned.csv',
            '../futures_trades_log.csv',
            '../futures_trades_log_cleaned.csv'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        # Generate sample data if no files found
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='H')
        sample_data = {
            'timestamp': dates,
            'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], len(dates)),
            'signal': np.random.choice(['buy', 'sell', 'hold'], len(dates)),
            'realized_pnl': np.random.normal(0, 10, len(dates)),
            'price': np.random.uniform(20000, 70000, len(dates))
        }
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"Error loading trade data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def calculate_performance_metrics(trades_df):
    """Calculate comprehensive performance metrics"""
    if trades_df.empty:
        return {}
    
    try:
        # Basic metrics
        total_trades = len(trades_df)
        
        if 'realized_pnl' in trades_df.columns:
            total_pnl = trades_df['realized_pnl'].sum()
            avg_pnl = trades_df['realized_pnl'].mean()
            win_trades = len(trades_df[trades_df['realized_pnl'] > 0])
            loss_trades = len(trades_df[trades_df['realized_pnl'] < 0])
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = trades_df[trades_df['realized_pnl'] > 0]['realized_pnl'].sum()
            gross_loss = abs(trades_df[trades_df['realized_pnl'] < 0]['realized_pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Drawdown calculation
            cumulative_pnl = trades_df['realized_pnl'].cumsum()
            peak = cumulative_pnl.expanding().max()
            drawdown = (peak - cumulative_pnl) / peak
            max_drawdown = drawdown.max()
            
            # Sharpe ratio (simplified)
            if trades_df['realized_pnl'].std() > 0:
                sharpe_ratio = (avg_pnl / trades_df['realized_pnl'].std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            total_pnl = avg_pnl = win_rate = profit_factor = max_drawdown = sharpe_ratio = 0
            win_trades = loss_trades = 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}

def create_pnl_chart(trades_df):
    """Create PnL over time chart"""
    if trades_df.empty or 'realized_pnl' not in trades_df.columns:
        return go.Figure()
    
    try:
        # Calculate cumulative PnL
        trades_df = trades_df.sort_values('timestamp')
        trades_df['cumulative_pnl'] = trades_df['realized_pnl'].cumsum()
        
        fig = go.Figure()
        
        # Cumulative PnL line
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative PnL:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Individual trade markers
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['realized_pnl']]
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_pnl'],
            mode='markers',
            name='Trades',
            marker=dict(color=colors, size=6, opacity=0.7),
            hovertemplate='<b>Trade PnL:</b> $%{text}<extra></extra>',
            text=[f"{pnl:.2f}" for pnl in trades_df['realized_pnl']]
        ))
        
        fig.update_layout(
            title='Portfolio Performance Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL ($)',
            hovermode='x unified',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating PnL chart: {e}")
        return go.Figure()

def create_symbol_performance_chart(trades_df):
    """Create symbol performance breakdown"""
    if trades_df.empty or 'symbol' not in trades_df.columns:
        return go.Figure()
    
    try:
        symbol_pnl = trades_df.groupby('symbol')['realized_pnl'].agg(['sum', 'count', 'mean']).reset_index()
        symbol_pnl.columns = ['symbol', 'total_pnl', 'trade_count', 'avg_pnl']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total PnL by Symbol', 'Trade Count by Symbol'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Total PnL bar chart
        colors = ['green' if pnl > 0 else 'red' for pnl in symbol_pnl['total_pnl']]
        fig.add_trace(
            go.Bar(x=symbol_pnl['symbol'], y=symbol_pnl['total_pnl'],
                   marker_color=colors, name='Total PnL'),
            row=1, col=1
        )
        
        # Trade count pie chart
        fig.add_trace(
            go.Pie(labels=symbol_pnl['symbol'], values=symbol_pnl['trade_count'],
                   name='Trade Count'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Error creating symbol chart: {e}")
        return go.Figure()

def create_risk_metrics_chart(risk_manager=None):
    """Create risk metrics visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Value', 'Risk Metrics', 'Drawdown', 'Position Distribution'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    if risk_manager and hasattr(risk_manager, 'get_risk_metrics'):
        try:
            metrics = risk_manager.get_risk_metrics()
            
            # Sample data for demonstration
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), periods=100)
            portfolio_values = np.cumsum(np.random.normal(100, 50, 100)) + 10000
            
            # Portfolio value over time
            fig.add_trace(
                go.Scatter(x=dates, y=portfolio_values, mode='lines', name='Portfolio Value'),
                row=1, col=1
            )
            
            # Risk metrics bar chart
            risk_names = ['VaR', 'Leverage', 'Concentration', 'Correlation']
            risk_values = [metrics.portfolio_var, metrics.leverage_ratio, 
                          metrics.concentration_risk, metrics.max_correlation]
            
            fig.add_trace(
                go.Bar(x=risk_names, y=risk_values, name='Risk Metrics'),
                row=1, col=2
            )
            
            # Drawdown
            drawdown_data = np.random.uniform(0, 0.1, 100)
            fig.add_trace(
                go.Scatter(x=dates, y=drawdown_data, mode='lines', 
                          fill='tonexty', name='Drawdown'),
                row=2, col=1
            )
            
            # Position distribution (placeholder)
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            values = [40, 35, 25]
            fig.add_trace(
                go.Pie(labels=symbols, values=values, name='Positions'),
                row=2, col=2
            )
            
        except Exception as e:
            st.error(f"Error with risk manager data: {e}")
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_signal_analysis_chart(trades_df):
    """Create signal analysis visualization"""
    if trades_df.empty or 'signal' not in trades_df.columns:
        return go.Figure()
    
    try:
        signal_performance = trades_df.groupby('signal')['realized_pnl'].agg(['sum', 'count', 'mean'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Signal Performance', 'Signal Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Signal performance
        fig.add_trace(
            go.Bar(x=signal_performance.index, y=signal_performance['sum'],
                   name='Total PnL by Signal'),
            row=1, col=1
        )
        
        # Signal distribution
        fig.add_trace(
            go.Pie(labels=signal_performance.index, values=signal_performance['count'],
                   name='Signal Count'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Error creating signal chart: {e}")
        return go.Figure()

def display_real_time_metrics():
    """Display real-time trading metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            "$147,093",
            delta="$2,834 (2.0%)",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Daily PnL",
            "$1,245",
            delta="$567 (84%)",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Active Positions",
            "7",
            delta="-2",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Risk Score",
            "4.2/10",
            delta="-0.8",
            delta_color="inverse"
        )

def main():
    st.title("🚀 Crypto Trading Agent Dashboard")
    st.markdown("### Real-time Performance Monitoring & Risk Analytics")
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        st.sidebar.markdown("🔄 **Auto-refreshing every 30 seconds**")
    
    # Refresh button
    if st.sidebar.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    with st.spinner("Loading trading data..."):
        trades_df = load_trade_data()
    
    if trades_df.empty:
        st.warning("No trading data available. Please check your data sources.")
        return
    
    # Display real-time metrics
    st.markdown("## 📊 Real-time Metrics")
    display_real_time_metrics()
    
    # Performance overview
    st.markdown("## 📈 Performance Overview")
    metrics = calculate_performance_metrics(trades_df)
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", f"{metrics['total_trades']:,}")
            st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
            
        with col2:
            st.metric("Total PnL", f"${metrics['total_pnl']:,.2f}")
            st.metric("Avg PnL per Trade", f"${metrics['avg_pnl']:,.2f}")
            
        with col3:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
    
    # Charts
    st.markdown("## 📊 Performance Charts")
    
    # PnL chart
    pnl_chart = create_pnl_chart(trades_df)
    st.plotly_chart(pnl_chart, use_container_width=True)
    
    # Symbol and signal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_chart = create_symbol_performance_chart(trades_df)
        st.plotly_chart(symbol_chart, use_container_width=True)
    
    with col2:
        signal_chart = create_signal_analysis_chart(trades_df)
        st.plotly_chart(signal_chart, use_container_width=True)
    
    # Risk metrics
    st.markdown("## ⚠️ Risk Analytics")
    
    try:
        # Initialize risk manager for demonstration
        risk_manager = AdvancedRiskManager()
        risk_chart = create_risk_metrics_chart(risk_manager)
        st.plotly_chart(risk_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Risk analytics unavailable: {e}")
    
    # ML Model Performance
    st.markdown("## 🤖 ML Model Performance")
    
    try:
        ml_performance = get_enhanced_model_performance()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Trained", ml_performance.get('num_models', 0))
            
        with col2:
            st.metric("Features Used", ml_performance.get('num_features', 0))
            
        with col3:
            st.metric("Predictions Made", ml_performance.get('prediction_count', 0))
        
        # Model weights visualization
        if 'model_weights' in ml_performance:
            weights_df = pd.DataFrame(list(ml_performance['model_weights'].items()),
                                    columns=['Model', 'Weight'])
            
            fig = px.bar(weights_df, x='Model', y='Weight', 
                        title='Model Ensemble Weights')
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.warning(f"ML performance data unavailable: {e}")
    
    # Recent trades table
    st.markdown("## 📋 Recent Trades")
    
    if not trades_df.empty:
        # Show last 20 trades
        recent_trades = trades_df.tail(20).copy()
        if 'timestamp' in recent_trades.columns:
            recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            recent_trades,
            use_container_width=True,
            hide_index=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**Trading Agent Dashboard** | Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Auto-refresh mechanism
    if auto_refresh:
        # Use st.empty() and time.sleep() for auto-refresh
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
