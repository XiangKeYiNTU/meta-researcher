# import os
import re
# import json
# from openai import OpenAI
# from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from tree_search.schemas import Plan, Step, ModificationResponse, PlanScore, ActionParams
# from schemas import Plan, Step, ModificationResponse, PlanScore
# from tree_search.prompts import (
#     initial_planning_prompt,
#     expand_prompt,
#     evaluate_plan_prompt
# )

# from tree_search.prompts import (
#     initial_planning_prompt,
#     expand_prompt,
#     evaluate_plan_prompt
# )

import base64
# import xmltodict

# DIRECT_UPLOAD_SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".json", ".jsonl", ".csv", ".doc", ".docx", ".pptx", ".py"]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def extract_plan(response: str) -> Tuple[str, Optional[Plan]]:
    # Find all <goal>...</goal> and <instruct>...</instruct> pairs
    goals = re.findall(r"<goal>(.*?)</goal>", response, re.DOTALL)
    instructs = re.findall(r"<instruct>(.*?)</instruct>", response, re.DOTALL)

    if not goals and not instructs:
        return "No <goal> or <instruct> tags found in the response.", None
    if not goals:
        return "No <goal> tags found in the response.", None
    if not instructs:
        return "No <instruct> tags found in the response.", None
    if len(goals) != len(instructs):
        return "Mismatched number of <goal> and <instruct> tags.", None

    steps = [Step(goal=goal.strip(), instructions=inst.strip()) for goal, inst in zip(goals, instructs)]
    return "success", Plan(steps=steps)

def extract_modification(response: str) -> Tuple[str, Optional[ModificationResponse]]:
    match  = re.search(r"<action>(.*?)</action>", response, re.DOTALL)

    if match:
         action = match.group(1).strip()
         if action == "add" or action == "update":
             position = re.search(r"<position>(.*?)</position>", response, re.DOTALL)
             goal = re.search(r"<goal>(.*?)</goal>", response, re.DOTALL)
             instruction = re.search(r"<instruct>(.*?)</instruct>", response, re.DOTALL)
             rationale = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
             if not position or not goal or not instruction:
                 return "Action parameters missing", None
             position = int(position.group(1).strip())
             goal = goal.group(1).strip()
             instruction = instruction.group(1).strip()
             if rationale:
                 rationale = rationale.group(1).strip()
             else:
                rationale = ""
             return "success", ModificationResponse(rationale=rationale, action=action, action_params=ActionParams(position=position, goal=goal, instructions=instruction))
         elif action == "remove":
             position = re.search(r"<position>(.*?)</position>", response, re.DOTALL)
             rationale = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
             if rationale:
                 rationale = rationale.group(1).strip()
             else:
                rationale = ""
             if not position:
                 return "Action parameters missing", None
             position = int(position.group(1).strip())
             return "success", ModificationResponse(rationale=rationale, action="remove", action_params=ActionParams(position=position, goal="", instructions=""))
         elif action == "keep":
             rationale = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
             if rationale:
                 rationale = rationale.group(1).strip()
             else:
                rationale = ""
             return "success", ModificationResponse(rationale=rationale, action="keep", action_params=ActionParams(position=-1, goal="", instructions=""))
         else:
             return "Unknown action, action must be one of four given actions: add, update, remove, and keep", None

    else:
        return "No action detected", None
    

def extract_scores(response: str) -> Tuple[str, Optional[PlanScore]]:
        effectiveness = re.search(r"<eff>(.*?)</eff>", response, re.DOTALL)
        completeness = re.search(r"<com>(.*?)</com>", response, re.DOTALL)
        executability = re.search(r"<exe>(.*?)</exe>", response, re.DOTALL)

        if not effectiveness or not completeness or not executability:
            return "Scores are missing", None
        effectiveness = int(effectiveness.group(1).strip())
        completeness = int(completeness.group(1).strip())
        executability = int(executability.group(1).strip())
        return "success", PlanScore(effectiveness=effectiveness, completeness=completeness, executability=executability)
        




    
# def generate_structured_response(openai_client: OpenAI, user_prompt: str, schema: BaseModel, system_prompt: str, file_path: str = None, model: str = "gpt-4"):
#     """
#     Generates a structured response (plan) for a given question using OpenAI's API.

#     Parameters:
#         openai_client (OpenAI): Initialized OpenAI client.
#         user_prompt (str): The user prompt.
#         schema (BaseModel): The Pydantic model to validate the response against.
#         system_prompt (str): The system prompt to guide the model.
#         file_path (str): Optional path to a file to include in the request.
#         model (str): The OpenAI model to use (default: "gpt-4").

#     Returns:
#         Plan: A structured plan based on the question.
#     """

#     if file_path:
#         if file_path.endswith(".pdf"):
#         # if file_path.split(".")[-1] in DIRECT_UPLOAD_SUPPORTED_EXTENSIONS:
#             file = openai_client.files.create(
#                 file=open(file_path, "rb"),
#                 purpose="user_data"
#             )

#             response = openai_client.responses.create(
#                 model=model,
#                 input=[
#                     {
#                         "role": "system",
#                         "content": system_prompt,
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "input_file",
#                                 "file_id": file.id,
#                             },
#                             {
#                                 "type": "input_text",
#                                 "text": user_prompt,
#                             },
#                         ]
#                     }
#                 ]
#             )

#             text = response.output_text
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             if match:
#                 json_str = match.group()
#                 obj = json.loads(json_str)
#                 try:
#                     res = schema.model_validate(obj)
#                     return res
#                 except Exception as e:
#                     raise ValueError(f"Failed to validate schema: {e}")

#             else:
#                 raise ValueError("No valid schema found in the response, please try again.")
            
#         elif file_path.endswith(".jpg") or file_path.endswith(".png") or file_path.endswith(".jpeg"):
#             encoded_image = encode_image(file_path)

#             completion = openai_client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": system_prompt,
#                     },
#                     {
#                         "role": "user",
#                         "content": [
#                             { "type": "text", "text": user_prompt },
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/{file_path.split('.')[-1]};base64,{encoded_image}",
#                                 },
#                             },
#                         ],
#                     }
#                 ],
#             )

#             text = completion.choices[0].message.content
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             if match:
#                 json_str = match.group()
#                 obj = json.loads(json_str)
#                 try:
#                     res = schema.model_validate(obj)
#                     return res
#                 except Exception as e:
#                     raise ValueError(f"Failed to validate schema: {e}")

#             else:
#                 raise ValueError("No valid schema found in the response, please try again.")
            
#         elif file_path.endswith(".zip"):
            # from document_tools.document_parser import DocumentParser
            # parser = DocumentParser()
            # extracted_files = parser._unzip_file(file_path)

#             response = openai_client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"{user_prompt}\n\nZip content:\n{extracted_files}"}
#                 ],
#             )

#             text = response.choices[0].message.content
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             if match:
#                 json_str = match.group()
#                 obj = json.loads(json_str)
#                 try:
#                     res = schema.model_validate(obj)
#                     return res
#                 except Exception as e:
#                     raise ValueError(f"Failed to validate schema: {e}")

#             else:
#                 raise ValueError("No valid schema found in the response, please try again.")

#         elif file_path.split(".")[-1] in UTF8_EXTENSIONS:
#             # If the file is a UTF-8 encoded text file, read it and include its content in the prompt
#             with open(file_path, "r", encoding="utf-8") as f:
#                 file_content = f.read()
            
#             response = openai_client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"{user_prompt}\n\nFile content:\n{file_content}"}
#                 ],
#             )

#             text = response.choices[0].message.content
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             if match:
#                 json_str = match.group()
#                 obj = json.loads(json_str)
#                 try:
#                     res = schema.model_validate(obj)
#                     return res
#                 except Exception as e:
#                     raise ValueError(f"Failed to validate schema: {e}")

#             else:
#                 raise ValueError("No valid schema found in the response, please try again.")

#         else:
#             response = openai_client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"{user_prompt}\n\nFile path:\n{file_path}"}
#                 ],
#             )

#             text = response.choices[0].message.content
#             match = re.search(r'\{.*\}', text, re.DOTALL)
#             if match:
#                 json_str = match.group()
#                 obj = json.loads(json_str)
#                 try:
#                     res = schema.model_validate(obj)
#                     return res
#                 except Exception as e:
#                     raise ValueError(f"Failed to validate schema: {e}")

#             else:
#                 raise ValueError("No valid schema found in the response, please try again.")
        
#     else:
#         response = openai_client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#         )

#         text = response.choices[0].message.content
#         match = re.search(r'\{.*\}', text, re.DOTALL)
#         if match:
#             json_str = match.group()
#             obj = json.loads(json_str)
#             try:
#                 res = schema.model_validate(obj)
#                 return res
#             except Exception as e:
#                 raise ValueError(f"Failed to validate schema: {e}")

#         else:
#             raise ValueError("No valid schema found in the response, please try again.")


# def generate_initial_plan(
#     openai_client: OpenAI,
#     # prompt: str,
#     question: str,
#     file_path: str = None,
#     model: str = "gpt-4o-mini",
#     # temperature: float = 0.7,
# ) -> Plan:
#     # build up prompt
#     prompt = initial_planning_prompt

#     initial_plan = generate_structured_response(
#         openai_client=openai_client,
#         user_prompt=question,
#         schema=Plan,
#         system_prompt=prompt,
#         file_path=file_path,
#         model=model
#     )

#     return initial_plan

# def modify_plan(
#     openai_client: OpenAI,
#     question: str,
#     plan: Plan,
#     file_path: str = None,
#     model: str = "gpt-4o-mini",
# ) -> ModificationResponse:
#     # build up prompt
#     system_prompt = expand_prompt

#     user_prompt = f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"

#     modification_response = generate_structured_response(
#         openai_client=openai_client,
#         user_prompt=user_prompt,
#         schema=ModificationResponse,
#         system_prompt=system_prompt,
#         file_path=file_path,
#         model=model
#     )

#     return modification_response


        
# def evaluate_plan(
#     openai_client: OpenAI,
#     question: str,
#     plan: Plan,
#     file_path: str = None,
#     model: str = "gpt-4o-mini",
# ) -> PlanScore:
#     # build up prompt
#     system_prompt = evaluate_plan_prompt

#     user_prompt = f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"

#     plan_score = generate_structured_response(
#         openai_client=openai_client,
#         user_prompt=user_prompt,
#         schema=PlanScore,
#         system_prompt=system_prompt,
#         file_path=file_path,
#         model=model
#     )

#     return plan_score

    # # upload file if provided
    # if file_path:
    #     file = openai_client.files.create(
    #         file=open(file_path, "rb"),
    #         purpose="user_data"
    #     )

    #     response = openai_client.responses.create(
    #         model=model,
    #         input=[
    #             {
    #                 "role": "system",
    #                 "content": prompt,
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "input_file",
    #                         "file_id": file.id,
    #                     },
    #                     {
    #                         "type": "input_text",
    #                         "text": f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}",
    #                     },
    #                 ]
    #             }
    #         ]
    #     )

    #     text = response.output_text
    #     match = re.search(r'\{.*\}', text, re.DOTALL)
    #     if match:
    #         json_str = match.group()
    #         score_obj = json.loads(json_str)
    #         try:
    #             score = PlanScore.model_validate(score_obj)
    #             return score
    #         except Exception as e:
    #             raise ValueError(f"Failed to validate plan score: {e}")
            
    #     else:
    #         raise ValueError("No valid plan score found in the response, please try again.")
    
    # else:
    #     response = openai_client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": prompt},
    #             {"role": "user", "content": f"Question: {question}\nPlan: {plan.model_dump_json(indent=2)}"},
    #         ],
    #     )

    #     text = response.choices[0].message.content
    #     match = re.search(r'\{.*\}', text, re.DOTALL)
    #     if match:
    #         json_str = match.group()
    #         score_obj = json.loads(json_str)
    #         try:
    #             score = PlanScore.model_validate(score_obj)
    #             return score
    #         except Exception as e:
    #             raise ValueError(f"Failed to validate plan score: {e}")

    #     else:
    #         raise ValueError("No valid plan score found in the response, please try again.")