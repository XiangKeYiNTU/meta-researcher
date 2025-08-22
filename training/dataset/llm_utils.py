import re
import json

from openai import OpenAI
from prompts import planning_prompt
from schemas import Plan
from typing import Tuple, Optional

def extract_plan(response: str) -> Tuple[str, Optional[Plan]]:
    # Find all substrings that *look like* JSON objects/arrays
    candidates = re.findall(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)

    longest_json = None
    max_length = 0

    for cand in candidates:
        try:
            json.loads(cand)  # check if it's valid JSON
            if len(cand) > max_length:
                longest_json = cand
                max_length = len(cand)
        except json.JSONDecodeError:
            continue
    if longest_json:
        try:
            obj = json.loads(longest_json)
            plan = Plan.model_validate(obj=obj)
            return "success", plan
        except:
            return "Invalid plan format", None
    else:
        return "No plans detected", None
    

def generate_plan(client: OpenAI, question: str, answer: str, model: str = "deepseek/deepseek-r1-0528:free") -> Plan:
    messages = [
        {
            "role": "system",
            "content": planning_prompt.format(question, answer)
        }
    ]
    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )

        message, plan = extract_plan(completion.choices[0].message.content)

        if message == "success":
            return plan
        else:
            messages.append(
                {
                    "role": "system",
                    "content": message
                }
            )
            continue
        

