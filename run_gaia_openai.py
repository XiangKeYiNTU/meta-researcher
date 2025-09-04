import argparse
# import sys
import os
# from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tree_search')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_explorer')))

from tree_search.openai.meta_planning_runner import MetaPlanningRunner
from plan_merger.base import PlanGraph
from agents.openai.meta_agent import MetaAgent
from web_explorer.openai.step_executor import StepExecutor
from memory.memory_manager import MemoryManager


# from web_explorer.openai.plan_executor import PlanRunner

from datasets import load_dataset

import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with GAIA benchmark.")

    parser.add_argument("--level", type=str, default="1", help="The level of the GAIA benchmark.")
    parser.add_argument("--split", type=str, default="validation", help="The split of the GAIA benchmark.")
    parser.add_argument("--planner_model", type=str, default="gpt-4o-mini", help="The model to use for planning.")
    parser.add_argument("--meta_model", type=str, default="gpt-4o-mini", help="The model to use for meta reasoning.")
    parser.add_argument("--executor_model", type=str, default="gpt-4o-mini", help="The model to use for executing the plan.")

    args = parser.parse_args()

    # load the GAIA benchmark
    test_set = load_dataset("./dataset/GAIA/GAIA.py", name=f"2023_level{args.level}", data_dir=".", split=args.split, trust_remote_code=True)

    print(test_set[0].keys())
    result_json = []
    # Initialize client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qwen_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
    # Initialize memory
    memory = MemoryManager(memory=[], client=qwen_client)
    MAX_QUESTION = 10
    cur_test = 0
    for i, task in enumerate(test_set):
        if cur_test >= MAX_QUESTION:
            break
        if task['file_path'] == "":
            cur_test += 1
            # result = {"task_id": task['task_id'], "question": task['Question'], "file_path": task['file_path'], "step_by_step_results": []}
            result = {"task_id": task['task_id'], "question": task['Question'], "step_by_step_results": []}
            print(f"Running task {i+1}/{len(test_set)}: {task['task_id']}")
            question = task['Question']
            # file_path = task['file_path'] if task['file_path'] != "" else None
            file_path = None

            # run the meta planning
            runner = MetaPlanningRunner(question=question, file_path=file_path, model=args.planner_model, openai_client=openai_client)
            search_tree = runner.run()
            top_plans = search_tree.select_top_plans()
            result['top_plans'] = [plan.model_dump() for plan in top_plans]
            plan_graph = PlanGraph()
            plan_graph.add_plan_list(top_plans)
            result['mermaid_graph'] = plan_graph.get_mermaid()

            # Start the execution
            meta_agent = MetaAgent(plan_graph=plan_graph, question=question, openai_client=openai_client, model=args.meta_model)
            while True:
                next_step = meta_agent.generate_next_step()
                # Meta agent chooses to skip
                if not next_step:
                    continue
                if next_step.goal == "END":
                    # finalize answer
                    final_answer = meta_agent.finalize_answer()
                    print(f"Final answer: {final_answer}")
                    result['final_answer'] = final_answer
                    break

                print(f"Next step to execute: {next_step.goal}")
                finished_steps = meta_agent.plan_graph.get_current_exec_results()
                step_executor = StepExecutor(
                    question=question,
                    current_step=next_step,
                    openai_client=openai_client,
                    qwen_client=qwen_client,
                    finished_steps=finished_steps,
                    file_path=file_path,
                    model=args.executor_model
                )
                step_result = step_executor.run()
                # todo: verify step results by meta agent
                result['step_by_step_results'].append(step_result)

                 # Add execution experience into memory
                memory.add(question=result['question'], 
                           step=next_step.goal, 
                           actions=step_result['actions'], 
                           result=step_result['result'], 
                           reference=step_result['reference'])

                # update graph
                step_node = meta_agent.plan_graph.exist_step(step=next_step)
                step_node.execution_result = step_result['result']

            result_json.append(result)



    #     result["top_k_plans"] = [plan.model_dump() for plan in top_k_plans]
    #     print(f"Executing the top plan: \n{top_k_plans[0].model_dump_json(indent=2)}")

    #     # Execute the plan using PlanExecutor
    #     executor = PlanRunner(plan=top_k_plans[0], question=question, file_path=file_path)
    #     step_by_step_results = executor.run(model=args.executor_model)
    #     result["step_by_step_results"] = step_by_step_results
    #     result_json.append(result)

    # # Save the results to a JSON file
    with open(f"GAIA_level{args.level}_{args.split}_results_openai.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)

    # save current memory
    memory.serialize()

