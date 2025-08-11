"""
Flask Dashboard for Real-Time Analytics
"""
from flask import Flask, render_template_string
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    base_dir = os.path.join(os.path.dirname(__file__), '../..')
    trade_log_path = os.path.join(base_dir, 'trade_log.csv')
    futures_log_path = os.path.join(base_dir, 'futures_trades_log.csv')
    analytics_path = os.path.join(base_dir, 'analytics_summary.csv')
    trade_log = pd.read_csv(trade_log_path) if os.path.exists(trade_log_path) else pd.DataFrame()
    futures_log = pd.DataFrame()
    futures_cols = ['timestamp','symbol','signal','price','atr','target_qty','prev_pos','new_pos','side','order_id','realized_pnl','simulated']
    if os.path.exists(futures_log_path):
        try:
            # Try standard read
            futures_log = pd.read_csv(futures_log_path)
            # If missing simulated column, add
            if len(futures_log.columns) == 11:
                futures_log['simulated'] = False
            elif len(futures_log.columns) > len(futures_cols):
                futures_log = futures_log.iloc[:,:len(futures_cols)]
            futures_log.columns = futures_cols
        except Exception:
            try:
                futures_log = pd.read_csv(futures_log_path, header=0, names=futures_cols, engine='python', on_bad_lines='skip')
            except Exception:
                futures_log = pd.DataFrame(columns=futures_cols)
    analytics = pd.read_csv(analytics_path) if os.path.exists(analytics_path) else None
    # Attempt to pull gateway metrics if available
    metrics = {}
    try:
        from trading_bot.utils import order_execution as oe
        if getattr(oe, '_gateway', None) and hasattr(oe._gateway, 'get_metrics'):
            metrics = oe._gateway.get_metrics()
    except Exception:
        pass
    template = """
    <h1>Crypto Trading Agent Dashboard</h1>
    <h2>Gateway Metrics</h2>
    {% if metrics %}
        <ul>
        {% for k,v in metrics.items() %}<li>{{k}}: {{v}}</li>{% endfor %}
        </ul>
    {% else %}<p>No metrics available.</p>{% endif %}
    <h2>Recent Spot Trades</h2>
    {% if not trade_log.empty %}
        {{ trade_log.tail(20).to_html(index=False) | safe }}
    {% else %}
        <p>No spot trade_log.csv found.</p>
    {% endif %}
    <h2>Recent Futures Trades</h2>
    {% if not futures_log.empty %}
        {{ futures_log.tail(50).to_html(index=False) | safe }}
    {% else %}
        <p>No futures_trades_log.csv found.</p>
    {% endif %}
    <h2>Analytics Summary</h2>
    {% if analytics is not none %}
        {{ analytics.tail(20).to_html(index=False) | safe }}
    {% else %}
        <p>No analytics_summary.csv found.</p>
    {% endif %}
    """
    return render_template_string(template, trade_log=trade_log, futures_log=futures_log, analytics=analytics, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
