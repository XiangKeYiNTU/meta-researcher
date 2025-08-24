import re
from openai import OpenAI


OPENROUTER_API_KEY = ""

CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

# Compile regex once for efficiency
REFERENCE_PATTERN = re.compile(r"<reference>(.*?)</reference>", re.DOTALL)

def extract_step_result_and_reference(solution_str: str):
    references = REFERENCE_PATTERN.findall(solution_str)
    answer = result = None

    if "#### " in solution_str:   # check longer prefix first
        result = solution_str.split("#### ", 1)[-1]
    elif "### " in solution_str:
        answer = solution_str.split("### ", 1)[-1]

    return {
        "references": references or None,
        "answer": answer,
        "result": result,
    }

def factuality_check_by_llm_judge(goal, references, result):
    # Build up prompt
    judge_prompt = f"""You are a judge that determines if the provided solution achieves the given goal and sticks to the references.
Respond with only "true" if the solution achieves the goal and is consistent with the references, respond with only "false" if it does not.

Goal:
{goal}

Reference list:
{''.join(reference + '\n' for reference in references)}

Solution:
{result}
"""
    
    messages = [
                {
                "role": "user",
                "content": judge_prompt
                }
            ]
    
    while True:
        completion = CLIENT.chat.completions.create(
            
            model="deepseek/deepseek-chat-v3-0324:free",
            # model = model,
            messages=messages
        )

        res = completion.choices[0].message.content

        if "true" in res:
            return 1.0
        elif "false" in res:
            return 0.0
        else:
            messages.append({
                "role": "user",
                "content": "Please respond with only 'true' or 'false'."
            })
            continue


def correctness_check_by_llm_judge(question, ground_truth, answer):
    # Build up prompt
    judge_prompt = f"""You are a judge that determines if the provided solution is consistent with the ground truth answer of a question.
Respond with only "true" if the solution is consistent, respond with only "false" if it is not.

Question:
{question}

Ground truth answer:
{ground_truth}

Solution:
{answer}
"""
    
    messages = [
                {
                "role": "user",
                "content": judge_prompt
                }
            ]
    
    while True:
        completion = CLIENT.chat.completions.create(
            
            model="deepseek/deepseek-chat-v3-0324:free",
            # model = model,
            messages=messages
        )

        res = completion.choices[0].message.content

        if "true" in res:
            return 1.0
        elif "false" in res:
            return 0.0
        else:
            messages.append({
                "role": "user",
                "content": "Please respond with only 'true' or 'false'."
            })
            continue


def step_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    extracted_solution = extract_step_result_and_reference(solution_str)
    # check if answer is provided
    if extracted_solution['answer']:
        score = correctness_check_by_llm_judge(question=extra_info['question'], ground_truth=extra_info['answer'], answer=extracted_solution['answer'])
    elif extracted_solution['result'] and extracted_solution['references']:
        score = factuality_check_by_llm_judge(goal=extra_info['goal'], references=extracted_solution['references'], result=extracted_solution['result'])
    else:
        score = 0.0
    
    return score