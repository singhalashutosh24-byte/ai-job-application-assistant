import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()  # reads the .env file and loads variables into environment

api_key = os.getenv("TAVILY_API_KEY")

if api_key is None:
    print("ERROR: TAVILY_API_KEY not found. Check your .env file.")
else:
    print(f"API key loaded successfully (starts with: {api_key[:8]}...)")

    client = TavilyClient(api_key=api_key)
    response = client.search(query="Google software engineer interview process")

    print("\n=== SEARCH RESULTS ===")
    for result in response["results"][:3]:  # just show first 3 results
        print(f"\nTitle: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content snippet: {result['content'][:200]}...")