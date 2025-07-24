import os
import re
import json
from openai import OpenAI
from schemas import Plan, Step, ModificationResponse, PlanScore
from prompts import (
    initial_planning_prompt,
    expand_prompt,
    evaluate_plan_prompt
)


def generate_initial_plan(
    openai_client: OpenAI,
    # prompt: str,
    question: str,
    file_path: str = None,
    model: str = "gpt-4",
    # temperature: float = 0.7,
) -> Plan:
    # build up prompt
    prompt = initial_planning_prompt

    # upload file if provided
    if file_path:
        file = openai_client.files.create(
            file=open(file_path, "rb"),
            purpose="user_data"
        )

        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": question,
                        },
                    ]
                }
            ]
        )

        text = response.output_text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            plan_obj = json.loads(json_str)
            try:
                plan = Plan.model_validate(plan_obj)
                return plan
            except Exception as e:
                raise ValueError(f"Failed to validate plan: {e}")

        else:
            raise ValueError("No valid plan found in the response, please try again.")
    
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
        )

        text = response.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            plan_obj = json.loads(json_str)
            try:
                plan = Plan.model_validate(plan_obj)
                return plan
            except Exception as e:
                raise ValueError(f"Failed to validate plan: {e}")

        else:
            raise ValueError("No valid plan found in the response, please try again.")

def modify_plan(
    openai_client: OpenAI,
    question: str,
    plan: Plan,
    file_path: str = None,
    model: str = "gpt-4",
) -> ModificationResponse:
    # build up prompt
    prompt = expand_prompt

    # upload file if provided
    if file_path:
        file = openai_client.files.create(
            file=open(file_path, "rb"),
            purpose="user_data"
        )

        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}",
                        },
                    ]
                }
            ]
        )

        text = response.output_text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            modification_obj = json.loads(json_str)
            try:
                modification_response = ModificationResponse.model_validate(modification_obj)
                return modification_response
            except Exception as e:
                raise ValueError(f"Failed to validate modification response: {e}")

        else:
            raise ValueError("No valid modification response found in the response, please try again.")
    
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"},
            ],
        )

        text = response.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            modification_obj = json.loads(json_str)
            try:
                modification_response = ModificationResponse.model_validate(modification_obj)
                return modification_response
            except Exception as e:
                raise ValueError(f"Failed to validate modification response: {e}")

        else:
            raise ValueError("No valid modification response found in the response, please try again.")
        
def evaluate_plan(
    openai_client: OpenAI,
    question: str,
    plan: Plan,
    file_path: str = None,
    model: str = "gpt-4",
) -> PlanScore:
    # build up prompt
    prompt = evaluate_plan_prompt

    # upload file if provided
    if file_path:
        file = openai_client.files.create(
            file=open(file_path, "rb"),
            purpose="user_data"
        )

        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}",
                        },
                    ]
                }
            ]
        )

        text = response.output_text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            score_obj = json.loads(json_str)
            try:
                score = PlanScore.model_validate(score_obj)
                return score
            except Exception as e:
                raise ValueError(f"Failed to validate plan score: {e}")
            
        else:
            raise ValueError("No valid plan score found in the response, please try again.")
    
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"},
            ],
        )

        text = response.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            score_obj = json.loads(json_str)
            try:
                score = PlanScore.model_validate(score_obj)
                return score
            except Exception as e:
                raise ValueError(f"Failed to validate plan score: {e}")

        else:
            raise ValueError("No valid plan score found in the response, please try again.")