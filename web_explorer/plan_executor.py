import sys

from utils import *
from schemas import *
from prompts import *
from search_api import *
from visit_api import *




class PlanRunner:
    def __init__(self, plan_path: str, question: str, file_path: str = None):
        self.plan = load_plan(plan_path=plan_path)
        self.question = question
        self.file_path = file_path
        self.finished_steps = []

    def execute_one_step(self, openai_client: OpenAI, qwen_client: OpenAI, step: Step, model: str):
        system_prompt = system_prompt

        # build up user prompt
        previous_steps = ""
        if len(self.finished_step) == 0:
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

        while True:
            # get response
            response = openai_client.responses.create(
                model=model,
                input=input
            )

            text = response.output_text

            # execute action
            action = extract_action(text)
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
                                "search count": search_count}
                self.finished_steps.append(step_results)
                return step_results
            else:
                input.append({
                    "role": "user",
                    "content": "The response format is invalid, you must include the available action markers: <search>, <visit>, <extract>, <summary>"
                })
                continue

    def run(self, openai_client: OpenAI, qwen_client: OpenAI, model: str):
        for i, step in enumerate(self.plan.steps):
            print(f"Starting executing step {i}:\nStep goal: {step.goal}\nStep instructions: {step.instructions}")
            step_results = self.execute_one_step(openai_client=openai_client, qwen_client=qwen_client, step=step, model=model)
            print(f"Step {i} done:\nexecution result: {step_results["result"]}\nfound relevent info:\n{step_results["found relevant info"]}\nwith {step_results["search count"]} searches.")

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
        if self.file_path:
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

        print(f"The final answer: {response.output_text}")




if __name__ == "__main__":
    # Get the path to the parent folder
    parent_env_path = Path(__file__).resolve().parents[1] / ".env"

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

    question = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    plan_path = sys.argv[2] if len(sys.argv) else Path(__file__).resolve().parents[1] / "example_plan.json"
    file_path = sys.argv[3] if len(sys.argv) > 3 else None

    model = sys.argv[4] if len(sys.argv) > 4 else "gpt-4o"

    runner = PlanRunner(plan_path=plan_path, question=question, file_path=file_path)
    runner.run(openai_client=openai_client, qwen_client=qwen_client, model=model)