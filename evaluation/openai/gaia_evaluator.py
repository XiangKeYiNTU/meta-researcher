import json
from typing import List

from evaluation.prompts import qa_eval_prompt
from openai import OpenAI
from datasets import load_dataset

class GAIAEvaluator:
    def __init__(self, level: int, split: str, client:  OpenAI, evaluator_model: str = "qwen/qwen3-235b-a22b:free"):
        self.level = level
        self.split = split
        self.evaluator_model = evaluator_model
        self.client = client

    def evaluate_single_question(self, question: str, gt_answer: str, pred_answer: str):
        prompt = qa_eval_prompt.format(
            question=question,
            labeled_answer=gt_answer,
            pred_answer=pred_answer
        )

        while True:
            response = self.client.chat.completions.create(
                model=self.evaluator_model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            if "Correct" in response.choices[0].message.content:
                return True
            if "Incorrect" in response.choices[0].message.content:
                return False
            
    def evaluate_complete_result(self, result_json_path: str) -> List[bool]:
        # load dataset
        dataset = load_dataset("./dataset/GAIA/GAIA.py", name=f"2023_level{self.level}", data_dir=".", split=self.split, trust_remote_code=True)


        # load predictions
        with open(result_json_path, 'r') as f:
            result_json = json.load(f)

        result = []

        for item in result_json:
            task_id = item['task_id']
            question = item['question']
            print(f"Evaluating task {task_id}: {question}")
            
            # find gt answer based on task_id
            gt_answer = next((d['Final answer'] for d in dataset if d['task_id'] == task_id), None)
            if 'final_answer' in item.keys():
                flag = self.evaluate_single_question(question, gt_answer, item['final_answer'])
            else:
                continue
            # flag = self.evaluate_single_question(task_id, gt_answer, item['step_by_step_results'][-1]['final_answer'])
            if flag:
                print(f"Task {task_id} passed.")
            else:
                print(f"Task {task_id} failed.")

            result.append(flag)

        return result
