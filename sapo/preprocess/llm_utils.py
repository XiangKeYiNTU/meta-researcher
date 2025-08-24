import re
import json

from openai import OpenAI
from prompts import planning_prompt
from schemas import Plan
from typing import Tuple, Optional

def extract_plan(response: str) -> Tuple[str, Optional[Plan]]:
    # Find all substrings that *look like* JSON objects/arrays
    # candidates = re.findall(r'(\{.*?\})', response, re.DOTALL)
    first_brace = response.find('{')
    last_brace = response.rfind('}')
    
    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        plan_str = ""
    else:
        plan_str = response[first_brace:last_brace + 1]

    if plan_str == "":
        return "No plans detected", None
    else:
        try:
            obj = json.loads(plan_str)
            plan = Plan.model_validate(obj=obj)
            return "success", plan
        except:
            return "Invalid plan format", None
        
def extract_ans_qwen(response: str):
    return response.split('</think>')[-1]
    

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

        # print(f"Model response: {completion.choices[0].message.content}")

        if model == "qwen/qwen3-235b-a22b:free":
            answer = extract_ans_qwen(completion.choices[0].message.content)
            message, plan = extract_plan(answer)
        else:
            message, plan = extract_plan(completion.choices[0].message.content)
        print(f"Extraction result: {message}")

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
        

