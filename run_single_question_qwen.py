import os
from openai import OpenAI
from transformers import pipeline


from tree_search.qwen.meta_tree_search_runner import MetaPlanner


# from web_explorer.openai.plan_executor import PlanRunner
from web_explorer.qwen.step_executor import StepExecutor
from plan_merger.base import PlanGraph
from agents.qwen.meta_agent import MetaAgent


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with a question.")
    parser.add_argument("question", type=str, help="The question to be answered by the planning runner.")
    parser.add_argument("--file_path", type=str, default=None, help="Optional file path for additional context.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-32B", help="Model ID or path.")

    args = parser.parse_args()

    qwen_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )

    generator = pipeline(
        "text-generation", 
        args.model_name_or_path, 
        torch_dtype="auto", 
        device_map="auto",
    )

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
        if "Final answer: " in step_result['result']:
            final_answer = step_result['result'].split("Final answer: ")[1].strip()
            print(f"Final answer: {final_answer}")
            break
        # update graph
        step_node = meta_agent.plan_graph.exist_step(step=next_step)
        step_node.execution_result = step_result['result']
