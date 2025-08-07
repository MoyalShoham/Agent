from openai import OpenAI

def analyze_market_data(research_data, social_data):
    # This is a placeholder for LLM logic
    # You can use OpenAI, Anthropic, or any LLM you prefer
    # Here, just a simple rule-based example:
    for coin in research_data:
        if "bitcoin" in coin:
            decision = "BUY"
        else:
            decision = "HOLD"
        print(f"Decision for {coin}: {decision}")
    return {"decision": decision, "coin": coin}