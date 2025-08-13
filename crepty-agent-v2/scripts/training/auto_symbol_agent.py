import os
import time
import requests
import json
from dotenv import load_dotenv
from loguru import logger
import openai
import threading

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com"

openai.api_key = OPENAI_API_KEY

AGENT_PROMPT = """
you are an expert broker, especially in crypto currencies
track the relevant data about the coin and let me know if it good
"""

# --- Helper functions ---
def get_hot_binance_futures_symbols():
    # Get top volume futures symbols from Binance
    url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        # Filter for high volume, low price, altcoins, micros, etc.
        filtered = [
            d for d in data
            if float(d.get("quoteVolume", 0)) > 10000000 and float(d.get("lastPrice", 0)) < 5 and d["symbol"].endswith("USDT")
        ]
        # Sort by volume descending
        filtered.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [d["symbol"] for d in filtered[:20]]
    except Exception as e:
        logger.error(f"Failed to fetch Binance futures symbols: {e}")
        return []

def get_coingecko_data(symbol):
    # Try to map Binance symbol to CoinGecko id
    try:
        # Remove USDT and lowercase
        coin = symbol[:-4].lower()
        url = f"{COINGECKO_API}/coins/markets?vs_currency=usd&ids={coin}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data:
            return data[0]
    except Exception as e:
        logger.warning(f"CoinGecko data failed for {symbol}: {e}")
    return {}

def get_twitter_sentiment(symbol):
    # Placeholder: In production, use Twitter API or scraping
    # Here, just return neutral
    return {"sentiment": "neutral", "mentions": 0}

def ai_should_add(symbol, binance_data, coingecko_data, twitter_data):
    # Compose context for OpenAI
    context = f"""
Symbol: {symbol}
Binance: {json.dumps(binance_data)}
CoinGecko: {json.dumps(coingecko_data)}
Twitter: {json.dumps(twitter_data)}
"""
    prompt = AGENT_PROMPT + "\n" + context + "\nShould this coin be added to the trading list? Answer yes or no and explain."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": context}],
            max_tokens=100,
            temperature=0.2
        )
        answer = response["choices"][0]["message"]["content"].lower()
        return "yes" in answer, answer
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return False, "AI error"

def update_futures_symbols(new_symbols):
    # Update the .env file FUTURES_SYMBOLS variable
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(env_path, "w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("FUTURES_SYMBOLS="):
                    f.write(f"FUTURES_SYMBOLS={','.join(new_symbols)}\n")
                else:
                    f.write(line)
        logger.success(f"Updated FUTURES_SYMBOLS in .env: {new_symbols}")
    except Exception as e:
        logger.error(f"Failed to update .env: {e}")

def agent_loop():
    while True:
        logger.info("[Agent] Checking for hot Binance futures symbols...")
        hot_symbols = get_hot_binance_futures_symbols()
        logger.info(f"[Agent] Hot symbols: {hot_symbols}")
        approved = []
        for symbol in hot_symbols:
            binance_data = {"symbol": symbol}
            coingecko_data = get_coingecko_data(symbol)
            twitter_data = get_twitter_sentiment(symbol)
            should_add, reason = ai_should_add(symbol, binance_data, coingecko_data, twitter_data)
            logger.info(f"AI decision for {symbol}: {should_add} ({reason})")
            if should_add:
                approved.append(symbol)
        if approved:
            update_futures_symbols(approved)
        else:
            logger.info("No new symbols approved by AI.")
        time.sleep(300)  # 5 minutes

def start_agent():
    t = threading.Thread(target=agent_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    start_agent()
    while True:
        time.sleep(60)
