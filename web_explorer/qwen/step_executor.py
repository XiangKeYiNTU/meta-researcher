from tree_search.schemas import Step
from typing import List, Tuple
import json
from openai import OpenAI

from transformers import pipeline

from web_explorer.prompts import system_prompt
from web_explorer.utils import extract_action, truncate_markdown, summarize_web_content_by_qwen
from web_explorer.search_api import get_text_search_results
from web_explorer.visit_api import visit

from document_tools.document_parser import DocumentParser

class StepExecutor:
    def __init__(self, generator: pipeline, current_step: Step, question: str, qwen_client: OpenAI, finished_steps: List[Tuple[Step, str]] = None, file_path: str = None):
        self.generator = generator
        self.finished_steps = finished_steps
        self.current_step = current_step
        self.question = question
        self.file_path = file_path
        self.qwen_client = qwen_client

    def run(self):
        if not self.finished_steps:
            previous_steps = "You are executing the first step."
        else:
            previous_steps = ""
            for step, answer in self.finished_steps:
                previous_steps += f"Step: {step.goal}\nAnswer: {answer}\n\n"

        user_prompt = f"Question: {self.question}\n\nPrevious steps and results:\n{previous_steps}Current step: {self.current_step.goal}\n\nInstructions: {self.current_step.instructions}"

        # parse file content
        if self.file_path:
            parser = DocumentParser()
            file_content = parser.parse_file(file_path=self.file_path)
            user_prompt += f"\n\nFile content:\n{file_content}"

        # Start execution
        extracted_info = []
        search_count = 0
        actions = []
        search_cache = {}
        visit_cache = {}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        while True:
            messages = self.generator(messages, max_new_tokens=32768)[0]["generated_text"]
            response = messages[-1]["content"]
            # execute action
            action = extract_action(response)
            # actions.append(action)
            action_step = {"action": action[0], "param": action[1] if len(action) > 1 else None}
            if action[0] == "search":
                search_count += 1
                if search_count > 5:
                    user_message = {
                        "role": "user",
                        "content": "Search quota reached. Please provide a final answer using <summary>."
                    }
                    messages.append(user_message)
                    action_step["action_result"] = "search quota reached"
                    actions.append(action_step)
                    continue
                else:
                    if action[1] in search_cache:
                        search_results = search_cache[action[1]]
                        result_string = json.dumps(search_results, indent=2)
                        print("WARNING: repeated search")
                        user_prompt = "WARNING: repeated search\n" + "```search_results\n" + result_string + "\n```"
                    else:
                        search_results = get_text_search_results(action[1])
                        search_cache[action[1]] = search_results
                        result_string = json.dumps(search_results, indent=2)
                        user_prompt = "```search_results\n" + result_string + "\n```"
                    messages.append({
                        "role": "user",
                        "content": user_prompt
                    })
                    action_step["action_result"] = result_string
                    actions.append(action_step)
                    # search_cache[action[1]] = search_results
                    continue
            elif action[0] == "visit":
                if action[1] in visit_cache:
                    raw_content = visit_cache[action[1]]
                    print("WARNING: repeated visit")
                    short_content = truncate_markdown(raw_content, max_tokens=20000)
                    web_summary = summarize_web_content_by_qwen(
                        step.goal, short_content, self.qwen_client
                    )
                    user_prompt = "WARNING: repeated visit\n" + "Here's a summary of the requested website:\n```web_content\n" + str(web_summary) + "\n```"
                else:
                    raw_content = visit(action[1])
                    visit_cache[action[1]] = raw_content
                # raw_content = visit(action[1])
                    short_content = truncate_markdown(raw_content, max_tokens=20000)
                    web_summary = summarize_web_content_by_qwen(
                        step.goal, short_content, self.qwen_client
                    )
                    user_prompt = "Here's a summary of the requested website:\n```web_content\n" + str(web_summary) + "\n```"
                messages.append({
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
                messages.append({
                    "role": "user",
                    "content": user_message
                })
                action_step["action_result"] = extracted_info
                actions.append(action_step)
                continue
            elif action[0] == "summary":
                step_results = {"goal": step.goal,
                                "result": action[1],
                                "found relevant info": extracted_info,
                                "search count": search_count,
                                "actions": actions}
                # self.finished_steps.append(step_results)
                return step_results
            elif action[0] == "finalize":
                step_results = {"goal": step.goal,
                                "result": f"Final answer: {action[1]}",
                                "found relevant info": extracted_info,
                                "search count": search_count,
                                "actions": actions}
                # self.finished_steps.append(step_results)
                # print(f"Final answer: {action[1]}")
                return step_results
            else:
                messages.append({
                    "role": "user",
                    "content": "The response format is invalid, you must include the available action markers: <search>, <visit>, <extract>, <summary>"
                })
                continue

