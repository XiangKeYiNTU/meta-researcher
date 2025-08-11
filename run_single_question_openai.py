# import sys
import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tree_search')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_explorer')))
from openai import OpenAI


from tree_search.openai.meta_planning_runner import MetaPlanningRunner


# from web_explorer.openai.plan_executor import PlanRunner
from web_explorer.openai.step_executor import StepExecutor
from plan_merger.base import PlanGraph
from agents.openai.meta_agent import MetaAgent


import argparse



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with a question.")
    parser.add_argument("question", type=str, help="The question to be answered by the planning runner.")
    parser.add_argument("--file_path", type=str, default=None, help="Optional file path for additional context.")
    parser.add_argument("--planner_model", type=str, default="gpt-4o-mini", help="The model to use for planning.")
    parser.add_argument("--meta_model", type=str, default="gpt-4o-mini", help="The model to use for meta reasoning.")
    parser.add_argument("--executor_model", type=str, default="gpt-4o-mini", help="The model to use for executing the plan.")

    args = parser.parse_args()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qwen_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
    runner = MetaPlanningRunner(question=args.question, file_path=args.file_path, model=args.planner_model, openai_client=openai_client)
    search_tree = runner.run()
    top_plans = search_tree.select_top_plans()
    # result['top_plans'] = [plan.model_dump() for plan in top_plans]
    plan_graph = PlanGraph()
    plan_graph.add_plan_list(top_plans)

    # Start the execution
    meta_agent = MetaAgent(plan_graph=plan_graph, question=args.question, openai_client=openai_client, model=args.meta_model)
    while True:
        next_step = meta_agent.generate_next_step()
        if next_step.goal == "END":
            # finalize answer
            final_answer = meta_agent.finalize_answer()
            print(f"Final answer: {final_answer}")
            break

        print(f"Next step to execute: {next_step.goal}")
        finished_steps = meta_agent.plan_graph.get_current_exec_results()
        step_executor = StepExecutor(
            current_step=next_step,
            question=args.question,
            openai_client=openai_client,
            qwen_client=qwen_client,
            finished_steps=finished_steps,
            file_path=args.file_path,
            model=args.executor_model
        )
        step_result = step_executor.run()
        if "Final answer: " in step_result['result']:
            final_answer = step_result['result'].split("Final answer: ")[1].strip()
            print(f"Final answer: {final_answer}")
            break
        # update graph
        step_node = meta_agent.plan_graph.exist_step(step=next_step)
        step_node.execution_result = step_result['result']

    

    # # Initialize and run the Meta Planning Runner
    # runner = MetaPlanningRunner(question=args.question, file_path=args.file_path, model=args.planner_model)
    # top_k_plans = runner.run()

    # print(f"Executing the top plan: \n{top_k_plans[0].model_dump_json(indent=2)}")

    # # Execute the plan using PlanExecutor
    # executor = PlanRunner(plan=top_k_plans[0], question=args.question, file_path=args.file_path)
    # executor.run(model=args.executor_model)