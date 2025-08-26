import os
import time
from pathlib import Path

from serpapi import GoogleSearch

from dotenv import load_dotenv

# Get the path to the parent folder
parent_env_path = Path(__file__).resolve().parents[1] / ".env"

# Load the .env file from the parent folder
load_dotenv(dotenv_path=parent_env_path)

API_KEY = os.getenv("SERP_API_KEY")
# print(f"Using SERP API Key: {API_KEY}")
retry_attempt = 3

def get_text_search_results(query, num_results=10):
    params = {
        "engine": "google",
        "q": query,
        "api_key": API_KEY,
        "num": num_results,
    }
    for i in range(retry_attempt):
        try:
            search = GoogleSearch(params)
            # debug: 
            # print(search.get_dict())
            organic_results = search.get_dict().get("organic_results", [])
            if not organic_results:
                return (f"No results found for query: {query}")
            parsed_results = []
            for result in organic_results:
                parsed_result = {
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet"),
                    "displayed_link": result.get("displayed_link"),
                }
                parsed_results.append(parsed_result)
            return parsed_results
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retry_attempt - 1:
                time.sleep(2)
            else:
                print("All retries failed.")
                return ("Connection error to the search engine. Please try again later.")
            
def get_image_search_results(image_path, num_results=10):
    params = {
    "engine": "google_lens",
    "search_type": "all",
    "url": image_path,
    "api_key": API_KEY
    }

    for i in range(retry_attempt):
        try:
            search = GoogleSearch(params)
            organic_results = search.get_dict().get("visual_matches", [])
            if not organic_results:
                return (f"No results found for image: {image_path}")
            if len(organic_results) < num_results:
                return organic_results
            # Limit the number of results to num_results
            parsed_results = organic_results[:num_results]
            return parsed_results
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retry_attempt - 1:
                time.sleep(2)
            else:
                print("All retries failed.")
                return ("Connection error to the search engine. Please try again later.")