from dotenv import load_dotenv
import os
from pathlib import Path
# Get the path to the parent folder
parent_env_path = Path(__file__).resolve().parents[1] / ".env"

# Load the .env file from the parent folder
load_dotenv(dotenv_path=parent_env_path)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"Using OpenRouter API Key: {OPENROUTER_API_KEY}")

import json
from openai import OpenAI

import tiktoken

from crawl4ai import AsyncWebCrawler
import asyncio

from schemas import Plan, Step

def extract_action(response: str):
    if "<search>" in response:
        search_query = response.split("<search>")[1].split("</search>")[0]
        return ("search", search_query)
    elif "<visit>" in response:
        visit_link = response.split("<visit>")[1].split("</visit>")[0]
        return ("visit", visit_link)
    elif "<extract>" in response:
        extracted_info = response.split("<extract>")[1].split("</extract>")[0]
        return ("extract", extracted_info)
    elif "<summary>" in response:
        final_answer = response.split("<summary>")[1].split("</summary>")[0]
        return ("summary", final_answer)
    else:
        return "no action detected"


def truncate_markdown(markdown_text, max_tokens=20000, model="gpt-4o"):
    """
    Truncates the markdown content to fit within a max token limit for a given GPT model.

    Parameters:
        markdown_text (str): Full markdown content to truncate.
        max_tokens (int): Maximum number of tokens allowed (default: 8000).
        model (str): The model name for tiktoken tokenizer (e.g., 'gpt-4o').

    Returns:
        str: Truncated markdown content.
    """
    # Load the appropriate tokenizer
    enc = tiktoken.encoding_for_model(model)

    # Tokenize the entire input
    tokens = enc.encode(markdown_text)

    if len(tokens) <= max_tokens:
        return markdown_text  # No truncation needed

    # Truncate and decode back to string
    truncated = enc.decode(tokens[:max_tokens])
    
    # Optionally: make a cleaner cut (e.g., end at paragraph or sentence)
    last_paragraph_end = truncated.rfind("\n\n")
    if last_paragraph_end != -1:
        truncated = truncated[:last_paragraph_end] + "\n\n*...(truncated)*"

    return truncated

def summarize_web_content_by_qwen(topic, web_content, openrouter_client):
    """
    Summarizes web content using Qwen model.

    Parameters:
        web_content (str): The content to summarize.
        openrouter_client: Optional client for OpenRouter API.

    Returns:
        str: Summary of the web content.
    """
    # Placeholder for Qwen summarization logic
    # This should be replaced with actual Qwen API call or logic

    summarize_prompt = f"""Summarize the webpage content relevant to the topic '{topic}'.
```web_content
{web_content}
```
"""
    response = openrouter_client.chat.completions.create(
        model="qwen/qwen3-235b-a22b:free",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to summarize webpage content relevant to the given topic."},
            {"role": "user", "content": summarize_prompt}
        ],
        max_tokens=2000,  # Adjust as needed
    )
    # return response # debug
    return response.choices[0].message.content

def load_plan(plan_path: str):
    # Read from file
    with open("plans.json", "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    # load plan
    try:
        plan = Plan(**loaded_data)
        return plan
    except Exception:
        raise ValueError("Plan format not correct, load failed.")


# async def main():
#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=OPENROUTER_API_KEY,
#     )

#     async with AsyncWebCrawler() as crawler:
#         result = await crawler.arun(url="https://www.amazon.sg")
#         markdown = result.markdown
#         short_markdown = truncate_markdown(markdown, max_tokens=20000)
#         summary = summarize_web_content_by_qwen("cosmetics", short_markdown, client)
#         print(summary)

# if __name__ == "__main__":
#     asyncio.run(main())