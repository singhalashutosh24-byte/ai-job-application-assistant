# agents/search_provider.py

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_api_key = os.getenv("TAVILY_API_KEY")
_client = TavilyClient(api_key=_api_key) if _api_key else None


def web_search(query: str, max_results: int = 3) -> list[str]:
    """
    Returns a list of content snippets for the given search query.
    Currently uses Tavily; swap the implementation here later 
    (e.g. Google Custom Search API) without touching any agent code.
    """
    if _client is None:
        raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

    response = _client.search(query=query, max_results=max_results)

    snippets = []
    for result in response.get("results", []):
        snippets.append(result.get("content", ""))

    return snippets