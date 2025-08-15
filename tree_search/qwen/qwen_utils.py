from transformers import pipeline
# from pydantic import BaseModel
from typing import Optional, Tuple, Callable, Any

from tree_search.prompts import (
    initial_planning_prompt,
    expand_prompt,
    evaluate_plan_prompt
)

from tree_search.llm_utils import (
    extract_plan,
    extract_modification,
    extract_scores
)

from tree_search.schemas import Plan, ModificationResponse, PlanScore

def generate_structured_response(generator: pipeline, system_prompt: str, user_prompt: str, extract_func: Callable[[str], Tuple[str, Optional[Any]]]):
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    while True:
        messages = generator(messages, max_new_tokens=32768)[0]["generated_text"]
        # print(messages[-1]["content"])
        response = messages[-1]["content"]

        extracted_result = extract_func(response)
        extract_message = extracted_result[0]
        extracted_data = extracted_result[1]
        if extract_message == "success":
            # if isinstance(extracted_data, schema):
            return extracted_data
            # return False, "Schema mismatch: The extracted data does not match the expected schema."
        else:
            messages.append(
                {"role": "user", "content": f"Error: {extract_message}. Please rephrase your response."}
            )

def generate_initial_plan(generator: pipeline, question: str, file_path: str = None) -> Plan:
    # build up user prompt

    if not file_path:
        user_prompt = f"Question: {question}"
    else:
        user_prompt = f"Question: {question}\n\nProvided File: {file_path}"

    return generate_structured_response(
        generator=generator,
        system_prompt=initial_planning_prompt,
        user_prompt=user_prompt,
        extract_func=extract_plan
    )

def modify_plan(generator: pipeline, plan: Plan, question: str, file_path: str = None) -> ModificationResponse:
    # build up user prompt
    if file_path:
        user_prompt = f"Question: {question}\n\nProvided file: {file_path}"
    else:
        user_prompt = f"Question: {question}"

    user_prompt += f"\n\nPlan:\n{plan.model_dump_json(indent=2)}"

    return generate_structured_response(
        generator=generator,
        system_prompt=expand_prompt,
        user_prompt=user_prompt,
        extract_func=extract_modification
    )

def evaluate_plan(generator: pipeline, plan: Plan, question: str, file_path: str = None) -> PlanScore:
    # build up user prompt
    if file_path:
        user_prompt = f"Question: {question}\n\nProvided file: {file_path}"
    else:
        user_prompt = f"Question: {question}"

    user_prompt += f"\n\nPlan:\n{plan.model_dump_json(indent=2)}"

    return generate_structured_response(
        generator=generator,
        system_prompt=evaluate_plan_prompt,
        user_prompt=user_prompt,
        extract_func=extract_scores
    )


