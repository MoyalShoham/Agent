"""
Performance Monitoring & Auto-Shutdown Module
Monitors live performance and disables underperforming strategies.
"""
def should_disable_strategy(win_rate, pnl, min_win_rate=0.3, min_pnl=0):
    return win_rate < min_win_rate or pnl < min_pnl
