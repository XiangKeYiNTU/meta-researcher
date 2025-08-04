from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

from web_explorer.utils import extract_action, truncate_markdown, summarize_web_content_by_qwen
from web_explorer.schemas import Plan, Step
# import web_explorer.prompts
from web_explorer.search_api import get_text_search_results
from web_explorer.visit_api import visit

# from utils import extract_action, truncate_markdown, summarize_web_content_by_qwen
# from schemas import Plan, Step
# # import prompts
from web_explorer.prompts import system_prompt
# from search_api import get_text_search_results
# from visit_api import visit

from document_tools.document_parser import DocumentParser

import base64

# DIRECT_UPLOAD_SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx", ".json", ".pptx", ".py", ".md"]
UTF8_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".jsonl", "jsonld", ".xml", ".py"]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class PlanRunner:
    def __init__(self, plan: Plan, question: str, file_path: str = None):
        # self.plan = load_plan(plan_path=plan_path)
        self.plan = plan
        self.question = question
        self.file_path = file_path
        self.finished_steps = []
        self.document_parser = DocumentParser()

    def execute_one_step(self, openai_client: OpenAI, qwen_client: OpenAI, step: Step, model: str):

        # build up user prompt
        previous_steps = ""
        if len(self.finished_steps) == 0:
            previous_steps = "You are now executing the first step."
        else:
            for i, finished_step in enumerate(self.finished_steps):
                previous_steps += f"Step {i}\n"
                previous_steps += f"Goal: {finished_step["goal"]}\n"
                previous_steps += f"Result: {finished_step["result"]}\n\n"

        current_step = f"Step goal: {step.goal}\n"
        current_step += f"Instructions: {step.instructions}"

        initial_user_prompt = f"Question: {self.question}\n\n"
        initial_user_prompt += "Previous steps and results:\n"
        initial_user_prompt += previous_steps
        initial_user_prompt += "You current step:\n"
        initial_user_prompt += current_step

        extracted_info = []
        search_count = 0
        input = []
        if self.file_path:
            if self.file_path.endswith(".pdf"):
            # if self.file_path.split(".")[-1] in DIRECT_UPLOAD_SUPPORTED_EXTENSIONS:
                # Directly upload the file to OpenAI
                file = openai_client.files.create(
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
                                "text": initial_user_prompt,
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
                                "text": initial_user_prompt,
                            }
                        ]
                    }
                ]

            elif self.file_path.split(".")[-1] in UTF8_EXTENSIONS:
                # Read the file content
                with open(self.file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

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
                                "text": initial_user_prompt + f"\n\nFile content:\n{file_content}",
                            },
                        ]
                    }
                ]

            elif self.file_path.endswith(".xlsx"):
                # Parse the Excel file using DocumentParser
                excel_content = self.document_parser.parse_excel_using_pandas(self.file_path)
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
                                "text": initial_user_prompt + f"\n\nExcel content:\n{excel_content}",
                            },
                        ]
                    }
                ]

            elif self.file_path.endswith(".pptx"):
                # Parse the PPT file using DocumentParser
                ppt_content = self.document_parser.parse_ppt(self.file_path)
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
                                "text": initial_user_prompt + f"\n\nPPT content:\n{'\n'.join(ppt_content)}",
                            },
                        ]
                    }
                ]

            elif self.file_path.endswith(".zip"):
                # Parse the ZIP file using DocumentParser
                zip_content = self.document_parser.parse_zip(self.file_path)
                zip_content_str = "\n".join([f"{file}: {content}" for item in zip_content for file, content in item.items()])
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
                                "text": initial_user_prompt + f"\n\nZIP content:\n{zip_content_str}",
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
                            {
                                "type": "input_text",
                                "text": initial_user_prompt + f"\n\nFile path:\n{self.file_path}\nThis file type is not supported for direct upload. If you can't solve the question without the file, please use <finalize> to output an approximated answer.",
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
                            "text": initial_user_prompt,
                        }
                    ]
                }
            ]

        # Record actions taken
        actions = []
        while True:
            # get response
            response = openai_client.responses.create(
                model=model,
                input=input
            )

            text = response.output_text

            print(f"Response from model: {text}")

            # execute action
            action = extract_action(text)
            actions.append(action)
            if action[0] == "search":
                search_count += 1
                if search_count > 5:
                    user_message = {
                        "role": "user",
                        "content": "Search quota reached. Please provide a final answer using <summary>."
                    }
                    input.append(user_message)
                    continue
                else:
                    search_results = get_text_search_results(action[1])
                    result_string = json.dumps(search_results, indent=2)
                    input.append({
                        "role": "user",
                        "content": "```search_results\n" + result_string + "\n```"
                    })
                    continue
            elif action[0] == "visit":
                raw_content = visit(action[1])
                short_content = truncate_markdown(raw_content, max_tokens=20000)
                web_summary = summarize_web_content_by_qwen(
                    step.goal, short_content, qwen_client
                )
                input.append({
                    "role": "user",
                    "content": "Here's a summary of the requested website:\n```web_content\n" + str(web_summary) + "\n```"
                })
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
                continue
            elif action[0] == "summary":
                step_results = {"goal": step.goal,
                                "result": action[1],
                                "found relevant info": extracted_info,
                                "search count": search_count,
                                "actions": actions}
                self.finished_steps.append(step_results)
                return step_results
            elif action[0] == "finalize":
                step_results = {"goal": step.goal,
                                "result": f"Final answer: {action[1]}",
                                "found relevant info": extracted_info,
                                "search count": search_count,
                                "actions": actions}
                self.finished_steps.append(step_results)
                # print(f"Final answer: {action[1]}")
                return step_results
            else:
                input.append({
                    "role": "user",
                    "content": "The response format is invalid, you must include the available action markers: <search>, <visit>, <extract>, <summary>"
                })
                continue

    def run(self, model: str):
        # Initialize OpenAI client
        # Get the path to the parent folder
        parent_env_path = Path(__file__).resolve().parents[2] / ".env"
        # Load the .env file from the parent folder
        load_dotenv(dotenv_path=parent_env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        # Initialize OpenAI client
        openai_client = OpenAI(api_key=api_key)

        # Initialize Openrouter client
        qwen_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        final_answer = None
        step_by_step_results = []
        for i, step in enumerate(self.plan.steps):
            print(f"Starting executing step {i}:\nStep goal: {step.goal}\nStep instructions: {step.instructions}")
            step_results = self.execute_one_step(openai_client=openai_client, qwen_client=qwen_client, step=step, model=model)
            step_by_step_results.append(step_results)
            if "Final answer:" in step_results["result"]:
                print(f"During step {i}, the final answer is found, no need to continue.")
                print(f"Final answer: {step_results['result']}")
                final_answer = step_results["result"].split("Final answer: ")[-1].strip()
                # self.finished_steps.append(step_results)
                break
            print(f"Step {i} done:\nexecution result: {step_results["result"]}\nfound relevent info:\n{step_results["found relevant info"]}\nwith {step_results["search count"]} searches.")

        if final_answer:
            print(f"Final answer: {final_answer}")

        else:
            previous_steps = ""
            for i, finished_step in enumerate(self.finished_steps):
                previous_steps += f"Step {i}\n"
                previous_steps += f"Goal: {finished_step["goal"]}\n"
                previous_steps += f"Result: {finished_step["result"]}\n\n"


            finalize_answer_prompt = f"Question: {self.question}\n\n"
            finalize_answer_prompt += "Previous steps and results:\n"
            finalize_answer_prompt += previous_steps
            finalize_answer_prompt += "Please finalize the answer to the question according to the previous steps and their results:\n"

            print("\nFinalizing answer...")
            input = []
            if self.file_path.split(".")[-1] in DIRECT_UPLOAD_SUPPORTED_EXTENSIONS:
                file = openai_client.files.create(
                    file=open(self.file_path, "rb"),
                    purpose="user_data"
                )
                input = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": finalize_answer_prompt,
                        },
                    ]
                }]

            elif self.file_path.endswith(".jpg") or self.file_path.endswith(".png") or self.file_path.endswith(".jpeg"):
                encoded_image = encode_image(self.file_path)
                image_url = f"data:image/{self.file_path.split('.')[-1]};base64,{encoded_image}"

                input = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_url,
                        },
                        {
                            "type": "input_text",
                            "text": finalize_answer_prompt,
                        },
                    ]
                }]
            else:
                input = [{
                    "role": "user",
                    "content": [
                        # {
                        #     "type": "input_file",
                        #     "file_id": file.id,
                        # },
                        {
                            "type": "input_text",
                            "text": finalize_answer_prompt,
                        },
                    ]
                }]
            response = openai_client.responses.create(
                model=model,
                input=input
            )
            final_answer = response.output_text
            print(f"The final answer: {response.output_text}")

        step_by_step_results.append({"final_answer": final_answer})
        return step_by_step_results




if __name__ == "__main__":

    # question = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    # # plan_path = sys.argv[2] if len(sys.argv) else Path(__file__).resolve().parents[1] / "example_plan.json"
    # file_path = sys.argv[3] if len(sys.argv) > 3 else None

    # model = sys.argv[4] if len(sys.argv) > 4 else "gpt-4o"

    question = "One of the songs on the VNV Nation album Futureperfect has a non-English title. The title references another piece of music. Who composed it? Answer using the format First name Last name."
    plan = Plan(
        steps=[
            Step(
                goal="Find the song on the VNV Nation album Futureperfect with a non-English title.",
                instructions="Search for the VNV Nation album Futureperfect and identify the song with a non-English title."
            ),
            Step(
                goal="Identify the piece of music referenced by the song.",
                instructions="Find out which piece of music is referenced by the identified song."
            ),
            Step(
                goal="Determine who composed the referenced piece of music.",
                instructions="Search for the composer of the referenced piece of music."
            )
        ]
    )

    file_path = None  # or specify a file path if needed
    model = "gpt-4o-mini"

    runner = PlanRunner(plan=plan, question=question, file_path=file_path)
    runner.run(model=model)