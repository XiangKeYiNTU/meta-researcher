import time

from openai import OpenAI
from serpapi import GoogleSearch
import requests

# from dotenv import load_dotenv

# # Get the path to the parent folder
# parent_env_path = Path(__file__).resolve().parents[4] / ".env"

# # Load the .env file from the parent folder
# load_dotenv(dotenv_path=parent_env_path)

# API_KEY = os.getenv("SERP_API_KEY")
# print(f"Using SERP API Key: {API_KEY}")
retry_attempt = 3

def get_text_search_results(query, api_key, num_results=10):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
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
            
def visit(url: str, api_key: str):
    request_url = "https://r.jina.ai/" + url
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(request_url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        # return f"Website Visit Error: {response.status_code}"
        raise Exception(f"Website visit error: {response.status_code}")
    
def summarize_website(content: str, api_key: str, model: str, topic: str = None):
    # Truncate web content if too long
    if len(content) > 20000:
        content = content[0:20000] + "\nrest of content omitted due to length..."

    # Build up summarizing prompt
    if topic:
        prompt = f"""Summarize the websize content below regarding to the topic, include key URL links that may contain helpful or relevant information.
Topic: {topic}

Website full content:
{content}
"""
    else:
        prompt = f"""Summarize the websize content below, include key URL links that may contain helpful or relevant information.

Website full content:
{content}
"""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        
        # model="deepseek/deepseek-chat-v3-0324:free",
        model = model,
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content



