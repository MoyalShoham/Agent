from pycoingecko import CoinGeckoAPI

def run_research_agent():
    cg = CoinGeckoAPI()
    trending = cg.get_search_trending()
    coins = [item['item']['id'] for item in trending['coins']]
    print("Trending coins:", coins)
    return coins