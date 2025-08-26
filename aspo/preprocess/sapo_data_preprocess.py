import argparse
import os
import re
import json
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/scratch/meta-researcher/verl/sapo/dataset/processed")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # data_source = "openai/gsm8k"
    # dataset = datasets.load_dataset(data_source, "main")

    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    data_path = "~/scratch/meta-researcher/verl/sapo/dataset/processed/hotpot_data.json"

    with open(data_path, 'r') as f:
        raw = f.read()

    unprocessed_dataset = json.loads(raw)['data']

    random.shuffle(unprocessed_dataset)

    
    # Compute split index (70%)
    split_idx = int(0.7 * len(unprocessed_dataset))

    # Split into 70/30
    train_dataset = unprocessed_dataset[:split_idx]
    test_dataset = unprocessed_dataset[split_idx:]
    
    def flatten_datapoint(datapoint):
        flattened_stepset = []
        for i, plan in enumerate(datapoint['plans']):
            for j, step in plan['steps']:
                flattened = {
                    "question": datapoint['question'],
                    "answer": datapoint['gt_answer'],
                    "plan_index": i,
                    "step_index": j,
                    "goal": step['goal'],
                    "instructions": step['instructions']
                }
                flattened_stepset.append(flattened)

        return flattened_stepset

    flattened_trainset = []
    flattened_testset = []
    for task in train_dataset:
        flattened_trainset.extend(flatten_datapoint(task))
    for task in test_dataset:
        flattened_testset.extend(flatten_datapoint(task))

    system_prompt = """You are executing a step in solving a more complex task.
You are given:
- A question
- Previous step results if any

Instructions:
- Use web search, site visits, and extraction to complete the step's goal. 
- If you encounter relevant information, include the original information in tags `<reference>` and `</reference>` 
- If you already have enough info, provide your final response of the step after "#### ".  
- If you "happen to" have enough info for the question (it's not required to answer the whole question during step execution), provide your final answer to the question after "### "

**Available tools**:  
- Use `skip` if you can extract the current step result from previous steps.
- Use `search` to conduct web searches
- Use `visit` to visit URL links


"""

    step_prompt = """Question: {}

Current step (your task):
{}
"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['question']

            # answer_raw = example.pop("answer")
            # solution = extract_solution(answer_raw)
            solution = example['answer']
            step_desc = f"""Step goal: {example['goal']}
Step instructions: {example['instructions']}
"""
            data = {
                "data_source": 'sapo_qa_n_plans',
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": step_prompt.format(question, step_desc),
                    },
                ],
                "ability": "search",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": example['answer'],
                    "question": example['question'],
                    "goal": example['goal'],
                    "instructions": example['instructions'],
                    "current_max_reward": 0,
                    "execution_result": ""
                    # "need_tools_kwargs": True,
                    # "tools_kwargs": {
                    #     # "calc_gsm8k_reward": {
                    #     #     "create_kwargs": {"ground_truth": solution},
                    #     #     # "execute_kwargs": {},
                    #     #     # "calc_reward_kwargs": {},
                    #     #     # "release_kwargs": {},
                    #     # },
                    #     "search": {
                    #         "create_kwargs": {"search_keywords": solution},
                    #         "execute_kwargs": {},
                    #         "calc_reward_kwargs": {},
                    #         "release_kwargs": {},
                    #     },
                    # },
                    # "interaction_kwargs": {
                    #     "query": question,
                    #     "ground_truth": solution,
                    # },
                },
            }
            return data

        return process_fn

    train_dataset = flattened_trainset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = flattened_testset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
