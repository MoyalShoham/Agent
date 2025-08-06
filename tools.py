from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_community import GoogleSearchRun, GoogleSearchAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from dotenv import load_dotenv
import requests


load_dotenv()

search = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())

search_tool = Tool(
    name="search_web",
    func=search.run,
    description="search the web for information",
)

def get_instagram_profile():
    url = f"https://graph.instagram.com/{INSTAGRAM_USER_ID}"
    params = {
        "fields": "id,username,media_count,account_type",
        "access_token": INSTAGRAM_ACCESS_TOKEN
    }
    response = requests.get(url, params=params)
    return response.json()

instagram_tool = Tool(
    name="instagram_profile",
    func=lambda _: get_instagram_profile(),
    description="Get Instagram profile information for the authenticated user."
)