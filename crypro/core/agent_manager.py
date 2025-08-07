from crypro.agents.research_agent import run_research_agent
from crypro.agents.social_agent import run_social_agent
from crypro.agents.analysis_agent import analyze_market_data
from crypro.agents.trading_agent import run_trading_agent

def run_all_agents():
    print("🔍 Running Research Agent...")
    research_data = run_research_agent()
    print("🐦 Running Social Agent...")
    social_data = run_social_agent(research_data)
    print("🧠 Running Analysis Agent...")
    analysis_result = analyze_market_data(research_data, social_data)
    print("💸 Running Trading Agent...")
    trade_result = run_trading_agent(analysis_result)
    print("✅ Trade Result:", trade_result)