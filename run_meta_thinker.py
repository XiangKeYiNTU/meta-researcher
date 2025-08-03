import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tree_search')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_explorer')))

from tree_search.meta_planning_runner import MetaPlanningRunner


from web_explorer.plan_executor import PlanRunner


import argparse



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with a question.")
    parser.add_argument("question", type=str, help="The question to be answered by the planning runner.")
    parser.add_argument("--file_path", type=str, default=None, help="Optional file path for additional context.")
    parser.add_argument("--planner_model", type=str, default="gpt-4o-mini", help="The model to use for planning.")
    parser.add_argument("--executor_model", type=str, default="gpt-4o-mini", help="The model to use for executing the plan.")

    args = parser.parse_args()

    

    # Initialize and run the Meta Planning Runner
    runner = MetaPlanningRunner(question=args.question, file_path=args.file_path, model=args.planner_model)
    top_k_plans = runner.run()

    print(f"Executing the top plan: \n{top_k_plans[0].model_dump_json(indent=2)}")

    # Execute the plan using PlanExecutor
    executor = PlanRunner(plan=top_k_plans[0], question=args.question, file_path=args.file_path)
    executor.run(model=args.executor_model)