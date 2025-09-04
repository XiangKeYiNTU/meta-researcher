from tree_search.schemas import Step
from typing import List, Tuple
import json

from openai import OpenAI
# import base64

from web_explorer.prompts import system_prompt
from web_explorer.utils import encode_image, extract_action, truncate_markdown, summarize_web_content_by_qwen
from web_explorer.search_api import get_text_search_results
from web_explorer.visit_api import visit


class StepExecutor:
    def __init__(self, question: str, current_step: Step, openai_client: OpenAI, qwen_client: OpenAI, finished_steps: List[Tuple[Step, str]] = None, file_path: str = None, model: str = "gpt-4o-mini"):
        self.question = question
        self.finished_steps = finished_steps
        self.current_step = current_step
        self.file_path = file_path
        self.model = model
        self.openai_client = openai_client
        self.qwen_client = qwen_client

    def run(self):
        # build up user query
        if not self.finished_steps:
            previous_steps = "You are executing the first step."
        else:
            previous_steps = ""
            for step, answer in self.finished_steps:
                previous_steps += f"Step: {step.goal}\nAnswer: {answer}\n\n"

        user_query = f"Question: {self.question}\n\nPrevious steps and results:\n{previous_steps}Current step: {self.current_step.goal}\n\nInstructions: {self.current_step.instructions}\n\n"
        user_prompt += "Stick to the current step during execution, use <answer> immediately when you gather enough info for the step, don't rush to solve the whole question!"

        # Start execution
        extracted_info = []
        search_count = 0
        visit_count = 0
        if self.file_path:
            if self.file_path.endswith(".pdf"):
                # Directly upload the file to OpenAI
                file = self.openai_client.files.create(
                    file=open(self.file_path, "rb"),
                    purpose="user_data"
                )

                input = [
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
                                "text": user_query,
                            },
                        ]
                    }
                ]

            elif self.file_path.endswith(".jpg") or self.file_path.endswith(".png") or self.file_path.endswith(".jpeg"):
                encoded_image = encode_image(self.file_path)
                image_url = f"data:image/{self.file_path.split('.')[-1]};base64,{encoded_image}"

                input = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": image_url,
                            },
                            {
                                "type": "input_text",
                                "text": user_query,
                            }
                        ]
                    }
                ]

            else:
                from document_tools.document_parser import DocumentParser
                document_parser = DocumentParser()
                document_content = document_parser.parse_file(self.file_path)
                input = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_query + f"\n\nDocument content:\n{document_content}",
                            },
                        ]
                    }
                ]

        else:
            input = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        # {
                        #     "type": "input_file",
                        #     "file_id": file.id,
                        # },
                        {
                            "type": "input_text",
                            "text": user_query,
                        }
                    ]
                }
            ]

        # Record actions taken
        actions = []
        search_cache = {}
        visit_cache = {}
        while True:
            # get response
            response = self.openai_client.responses.create(
                model=self.model,
                input=input
            )

            text = response.output_text

            print(f"Response from model: {text}")

            # execute action
            action = extract_action(text)
            # actions.append(action)
            action_step = {"action": action[0], "param": action[1] if len(action) > 1 else None}
            if action[0] == "search":
                search_count += 1
                if search_count > 10:
                    user_message = {
                        "role": "user",
                        "content": "Search quota reached. Please provide a final answer using <answer>."
                    }
                    input.append(user_message)
                    action_step["action_result"] = "search quota reached"
                    actions.append(action_step)
                    continue
                else:
                    if action[1] in search_cache:
                        search_results = search_cache[action[1]]
                        result_string = json.dumps(search_results, indent=2)
                        user_prompt = "WARNING: repeated search\n" + "```search_results\n" + result_string + "\n```"
                    else:
                        search_results = get_text_search_results(action[1])
                        search_cache[action[1]] = search_results
                        result_string = json.dumps(search_results, indent=2)
                        user_prompt = "```search_results\n" + result_string + "\n```"
                    input.append({
                        "role": "user",
                        "content": user_prompt
                    })
                    action_step["action_result"] = result_string
                    actions.append(action_step)
                    # search_cache[action[1]] = search_results
                    continue
            elif action[0] == "visit":
                visit_count += 1
                if visit_count > 20:
                    user_message = {
                        "role": "user",
                        "content": "Visit quota reached. Please provide a final answer using <answer>."
                    }
                    input.append(user_message)
                    action_step["action_result"] = "visit quota reached"
                    actions.append(action_step)
                    continue
                topic = action[2]
                if not topic:
                    topic = self.current_step.goal
                if action[1] in visit_cache:
                    raw_content = visit_cache[action[1]]
                    # print("WARNING: repeated visit")
                    short_content = truncate_markdown(raw_content, max_tokens=20000)
                    web_summary = summarize_web_content_by_qwen(
                        topic, short_content, self.qwen_client
                    )
                    user_prompt = "WARNING: repeated visit\n" + "Here's a summary of the requested website:\n```web_content\n" + str(web_summary) + "\n```"
                else:
                    raw_content = visit(action[1])
                    visit_cache[action[1]] = raw_content
                # raw_content = visit(action[1])
                    short_content = truncate_markdown(raw_content, max_tokens=20000)
                    web_summary = summarize_web_content_by_qwen(
                        topic, short_content, self.qwen_client
                    )
                    user_prompt = "Here's a summary of the requested website:\n```web_content\n" + str(web_summary) + "\n```"
                input.append({
                    "role": "user",
                    "content": user_prompt
                })
                action_step["action_result"] = web_summary
                actions.append(action_step)
                continue
            elif action[0] == "extract":
                extracted_info.append(action[1])
                user_message = "Here are all the extracted information so far:\n"
                for info in extracted_info:
                    user_message += f"{info}\n"
                input.append({
                    "role": "user",
                    "content": user_message
                })
                action_step["action_result"] = extracted_info
                actions.append(action_step)
                continue
            elif action[0] == "summary":
                reference = action[2]
                step_results = {
                    "goal": self.current_step.goal,
                    "result": action[1],
                    "reference": reference if reference else "No reference provided",
                    "found_relevant_info": extracted_info,
                    "search_count": search_count,
                    "visit_count": visit_count,
                    "actions": actions
                }
                # self.finished_steps.append(step_results)
                return step_results
            # elif action[0] == "finalize":
            #     step_results = {"goal": self.current_step.goal,
            #                     "result": f"Final answer: {action[1]}",
            #                     "found relevant info": extracted_info,
            #                     "search count": search_count,
            #                     "actions": actions}
            #     # self.finished_steps.append(step_results)
            #     # print(f"Final answer: {action[1]}")
            #     return step_results
            else:
                input.append({
                    "role": "user",
                    "content": "The response format is invalid, you must include the available action markers: <search>, <visit>, <extract>, <answer>"
                })
                continue