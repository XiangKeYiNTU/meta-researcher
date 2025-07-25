import os
import requests
from pathlib import Path

from dotenv import load_dotenv

# Get the path to the parent folder
parent_env_path = Path(__file__).resolve().parents[1] / ".env"

# Load the .env file from the parent folder
load_dotenv(dotenv_path=parent_env_path)

API_KEY = os.getenv("JINA_API_KEY")

def visit(url: str):
    request_url = "https://r.jina.ai/" + url
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(request_url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        return f"Website Visit Error: {response.status_code}"
    
if __name__ == "__main__":
    # Example usage
    url = "https://www.amazon.sg/"
    result = visit(url)
    print(result)