import argparse
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tree_search')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_explorer')))

from tree_search.openai.meta_planning_runner import MetaPlanningRunner


from web_explorer.openai.plan_executor import PlanRunner

from datasets import load_dataset

import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with GAIA benchmark.")

    parser.add_argument("--level", type=str, default="1", help="The level of the GAIA benchmark.")
    parser.add_argument("--split", type=str, default="test", help="The split of the GAIA benchmark.")
    parser.add_argument("--planner_model", type=str, default="gpt-4o-mini", help="The model to use for planning.")
    parser.add_argument("--executor_model", type=str, default="gpt-4o-mini", help="The model to use for executing the plan.")

    args = parser.parse_args()

    # load the GAIA benchmark
    test_set = load_dataset(".\dataset\GAIA\GAIA.py", name=f"2023_level{args.level}", data_dir=".", split=args.split)

    # print(test_set[0].keys())
    result_json = []
    for i, task in enumerate(test_set):
        result = {"task_id": task['task_id'], "question": task['Question'], "file_path": task['file_path']}
        print(f"Running task {i+1}/{len(test_set)}: {task['task_id']}")
        question = task['Question']
        file_path = task['file_path'] if 'file_path' != "" else None
        
        # run the meta planning
        runner = MetaPlanningRunner(question=question, file_path=file_path, model=args.planner_model)
        top_k_plans = runner.run()
        result["top_k_plans"] = [plan.dict() for plan in top_k_plans]
        print(f"Executing the top plan: \n{top_k_plans[0].model_dump_json(indent=2)}")

        # Execute the plan using PlanExecutor
        executor = PlanRunner(plan=top_k_plans[0], question=question, file_path=file_path)
        step_by_step_results = executor.run(model=args.executor_model)
        result["step_by_step_results"] = step_by_step_results
        result_json.append(result)

    # Save the results to a JSON file
    with open(f"GAIA_level{args.level}_{args.split}_results.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)

