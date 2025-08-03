import os
# import sys
import argparse
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# from tree_search.base import InitialNode, ModifiedNode, BaseTreeNode, SearchTree
from base import InitialNode, ModifiedNode, BaseTreeNode, SearchTree
# from tree_search.llm_utils import (
#     generate_initial_plan,
#     modify_plan,
#     evaluate_plan
# )

from llm_utils import (
    generate_initial_plan,
    modify_plan,
    evaluate_plan
)



class MetaPlanningRunner:
    def __init__(self, question: str, file_path: str = None, model: str = "gpt-4"):
        self.question = question
        self.file_path = file_path
        self.model = model
    
    def run(self):
        # Get the path to the parent folder
        parent_env_path = Path(__file__).resolve().parents[1] / ".env"

        # Load the .env file from the parent folder
        load_dotenv(dotenv_path=parent_env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=api_key)


        # First create an initial plan
        print(f"Generating initial plan for question: {self.question}")
        initial_plan = generate_initial_plan(openai_client, self.question, self.file_path, self.model)

        # Evaluate the initial plan
        initial_score = evaluate_plan(openai_client, self.question, initial_plan, self.file_path, self.model)

        # Append the initial plan to the search tree
        print("Creating search tree with initial plan...")
        search_tree = SearchTree(question=self.question, initial_plan=initial_plan, initial_score=initial_score)

        print("Start expanding the search tree...")

        selected_node = search_tree.select()
        while selected_node:
            # Expand the selected node
            modifications = modify_plan(openai_client, self.question, selected_node.get_plan(), self.file_path, self.model)

            print(f"Modifications for node\n {selected_node.get_plan()}:\n {modifications}")

            # Create modified nodes and append them to the search tree
            expanded_node = ModifiedNode(
                parent=selected_node,
                rationale=modifications.rationale,
                action=modifications.action,
                action_params=modifications.action_params
            )

            selected_node.children.append(expanded_node)

            # Evaluate the modified plan
            plan = expanded_node.get_plan()
            expanded_node.score = evaluate_plan(openai_client, self.question, plan, self.file_path, self.model)


            # Select the next node for expansion
            selected_node = search_tree.select()

        # Print out the final search tree
        search_tree.print_tree()

        # Select the top k plans
        top_k_plans = search_tree.select_top_k(top_k=3)

        return top_k_plans

        # if save:
        #     plan_folder = Path(__file__).resolve().parent / "plans"
        #     plan_folder.mkdir(parents=True, exist_ok=True)

        #     save_path = plan_folder / f"top_3_{self.question.replace(' ', '_')}.json"
        #     search_tree.serialize(save_path, top_k=3)

if __name__ == "__main__":
    # question = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    # file_path = sys.argv[2] if len(sys.argv) > 2 else None
    # model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"

    arg_parser = argparse.ArgumentParser(description="Meta Planning Runner")
    arg_parser.add_argument("--question", type=str, required=True, help="The question to answer.")
    arg_parser.add_argument("--file_path", type=str, default=None, help="Path to the file to use for context.")
    arg_parser.add_argument("--model", type=str, default="gpt-4o-mini", help="The model to use for LLM operations.")
    # arg_parser.add_argument("--save", action='store_true', help="Whether to save the search tree to a file.")

    args = arg_parser.parse_args()

    # Initialize the MetaPlanningRunner with the provided arguments
    runner = MetaPlanningRunner(question=args.question, file_path=args.file_path, model=args.model)
    runner.run()