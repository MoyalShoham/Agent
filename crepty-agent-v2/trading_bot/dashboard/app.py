"""
Flask Dashboard for Real-Time Analytics
"""
from flask import Flask, render_template_string
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    trade_log = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../trade_log.csv'))
    analytics = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../analytics_summary.csv')) if os.path.exists(os.path.join(os.path.dirname(__file__), '../../analytics_summary.csv')) else None
    return render_template_string('''
    <h1>Crypto Trading Agent Dashboard</h1>
    <h2>Recent Trades</h2>
    {{ trade_log.tail(20).to_html() }}
    <h2>Analytics Summary</h2>
    {% if analytics is not none %}
        {{ analytics.tail(20).to_html() }}
    {% else %}
        <p>No analytics_summary.csv found.</p>
    {% endif %}
    ''', trade_log=trade_log, analytics=analytics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
