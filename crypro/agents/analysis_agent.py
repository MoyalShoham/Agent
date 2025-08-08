from openai import OpenAI

def analyze_market_data(research_data, social_data):
    # Placeholder: simple rule-based logic
    for coin in research_data:
        if "bitcoin" in coin:
            decision = "BUY"
        else:
            decision = "HOLD"
        print(f"Decision for {coin}: {decision}")
    return {"decision": decision, "coin": coin}