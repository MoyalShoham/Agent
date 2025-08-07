import snscrape.modules.twitter as sntwitter

def run_social_agent(coins):
    sentiment = {}
    for coin in coins:
        query = f"{coin} crypto"
        tweets = [tweet.content for tweet in sntwitter.TwitterSearchScraper(query).get_items()]
        sentiment[coin] = f"Found {len(tweets)} tweets for {coin}"  # Placeholder: add real sentiment analysis
    print("Social sentiment:", sentiment)
    return sentiment