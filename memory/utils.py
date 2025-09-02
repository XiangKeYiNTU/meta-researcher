import re
from openai import OpenAI

def extract_experiences(response: str):
    # Extract lines starting with a number followed by a dot
    lines = re.findall(r'^\d+\..*', response, re.MULTILINE)
    return lines

