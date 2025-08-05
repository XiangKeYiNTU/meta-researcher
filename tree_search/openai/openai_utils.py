from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Tuple, Callable, Any

from tree_search.prompts import (
    initial_planning_prompt,
    expand_prompt,
    evaluate_plan_prompt
)

from tree_search.llm_utils import (
    encode_image,
    extract_plan,
    extract_modification,
    extract_scores
)

from tree_search.schemas import Plan, ModificationResponse, PlanScore

UTF8_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".jsonl", "jsonld", ".xml", ".py"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def generate_structured_response(openai_client: OpenAI, user_prompt: str, schema: BaseModel, system_prompt: str, extract_func: Callable[[str], Tuple[str, Optional[Any]]], file_path: str = None, model: str = "gpt-4o-mini"):
    if not file_path:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # response = openai_client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ],
        # )

        # response_text = response.choices[0].message.content
        # extracted_result = extract_func(response_text)
    else:
        if file_path.split(".")[-1] in IMAGE_EXTENSIONS:
            encoded_image = encode_image(file_path)
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": user_prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/{file_path.split('.')[-1]};base64,{encoded_image}",
                        },
                    ],
                }
            ]
        elif file_path.split(".")[-1] in UTF8_EXTENSIONS:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nFile content:\n" + file_content}
            ]
        elif file_path.split(".")[-1] == "pdf":
            file = openai_client.files.create(
                file=open(file_path, "rb"),
                purpose="user_data"
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
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
                            "text": user_prompt,
                        },
                    ]
                }
            ]
        elif file_path.endswith(".zip"):
            from document_tools.document_parser import DocumentParser
            parser = DocumentParser()
            extracted_files = parser._unzip_file(file_path)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nZip content:\n{extracted_files}"}
            ]

        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + f"\n\nFile path: {file_path}"}
            ]

    while True:

        response = openai_client.responses.create(
            model=model,
            input=messages
        )

        response_text = response.output_text
        extracted_result = extract_func(response_text)
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

def generate_initial_plan(
    openai_client: OpenAI,
    # prompt: str,
    question: str,
    file_path: str = None,
    model: str = "gpt-4o-mini",
    # temperature: float = 0.7,
) -> Plan:
    # build up prompt
    prompt = initial_planning_prompt

    initial_plan = generate_structured_response(
        openai_client=openai_client,
        user_prompt=question,
        schema=Plan,
        system_prompt=prompt,
        extract_func=extract_plan,
        file_path=file_path,
        model=model
    )

    return initial_plan

def modify_plan(
    openai_client: OpenAI,
    question: str,
    plan: Plan,
    file_path: str = None,
    model: str = "gpt-4o-mini",
) -> ModificationResponse:
    # build up prompt
    system_prompt = expand_prompt

    user_prompt = f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"

    modification_response = generate_structured_response(
        openai_client=openai_client,
        user_prompt=user_prompt,
        schema=ModificationResponse,
        system_prompt=system_prompt,
        extract_func=extract_modification,
        file_path=file_path,
        model=model
    )

    return modification_response

def evaluate_plan(
    openai_client: OpenAI,
    question: str,
    plan: Plan,
    file_path: str = None,
    model: str = "gpt-4o-mini",
) -> PlanScore:
    # build up prompt
    system_prompt = evaluate_plan_prompt

    user_prompt = f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"

    plan_score = generate_structured_response(
        openai_client=openai_client,
        user_prompt=user_prompt,
        schema=PlanScore,
        system_prompt=system_prompt,
        extract_func=extract_scores,
        file_path=file_path,
        model=model
    )

    return plan_score