import os
from openai import OpenAI

import argparse

from transformers import pipeline

from tree_search.qwen.meta_tree_search_runner import MetaPlanner
from plan_merger.base import PlanGraph
from agents.qwen.meta_agent import MetaAgent
from web_explorer.qwen.step_executor import StepExecutor

from datasets import load_dataset

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with GAIA benchmark.")

    parser.add_argument("--level", type=str, default="1", help="The level of the GAIA benchmark.")
    parser.add_argument("--split", type=str, default="validation", help="The split of the GAIA benchmark.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-32B", help="Model ID or path.")

    args = parser.parse_args()

    generator = pipeline(
        "text-generation", 
        args.model_name_or_path, 
        torch_dtype="auto", 
        device_map="auto",
    )

    qwen_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )

    # load the GAIA benchmark
    test_set = load_dataset("./dataset/GAIA/GAIA.py", name=f"2023_level{args.level}", data_dir=".", split=args.split, trust_remote_code=True)

    # print(test_set[0].keys())
    result_json = []
    for i, task in enumerate(test_set):
        result = {"task_id": task['task_id'], "question": task['Question'], "file_path": task['file_path'], "step_by_step_results": []}
        print(f"Running task {i+1}/{len(test_set)}: {task['task_id']}")
        question = task['Question']
        file_path = task['file_path'] if task['file_path'] != "" else None

        plan_runner = MetaPlanner(generator=generator,question=args.question,file_path=args.file_path)

        search_tree = plan_runner.run()
        top_plans = search_tree.select_top_plans()
        # result['top_plans'] = [plan.model_dump() for plan in top_plans]
        plan_graph = PlanGraph()
        plan_graph.add_plan_list(top_plans)

        meta_agent = MetaAgent(plan_graph=plan_graph, question=args.question, generator=generator)

        while True:
            next_step = meta_agent.generate_next_step()
            if next_step.goal == "END":
                # finalize answer
                final_answer = meta_agent.finalize_answer()
                print(f"Final answer: {final_answer}")
                result['final_answer'] = final_answer
                break

            print(f"Next step to execute: {next_step.goal}")
            finished_steps = meta_agent.plan_graph.get_current_exec_results()
            step_executor = StepExecutor(
                generator=generator,
                current_step=next_step,
                question=args.question,
                finished_steps=finished_steps,
                file_path=args.file_path,
                qwen_client=qwen_client
            )

            step_result = step_executor.run()
            result['step_by_step_results'].append(step_result)
            if "Final answer: " in step_result['result']:
                final_answer = step_result['result'].split("Final answer: ")[1].strip()
                print(f"Final answer: {final_answer}")
                result['final_answer'] = final_answer
                break
            # update graph
            step_node = meta_agent.plan_graph.exist_step(step=next_step)
            step_node.execution_result = step_result['result']

        result_json.append(result)

    with open(f"GAIA_level{args.level}_{args.split}_qwen_results.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)