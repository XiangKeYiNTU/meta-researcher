import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

from base import InitialNode, ModifiedNode, BaseTreeNode, SearchTree
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

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    file_path = sys.argv[2] if len(sys.argv) > 2 else None
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"

    runner = MetaPlanningRunner(question=question, file_path=file_path, model=model)
    runner.run()