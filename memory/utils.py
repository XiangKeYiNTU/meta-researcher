import re
from openai import OpenAI

def extract_experiences(response: str):
    # Extract lines starting with a number followed by a dot
    lines = re.findall(r'^\d+\..*', response, re.MULTILINE)
    # Remove the prefixed number and dot with optional space
    clean_lines = [re.sub(r'^\d+\.\s*', '', line) for line in lines]
    return clean_lines

