import argparse
from transformers import pipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer

from tree_search.base import SearchTree, ModifiedNode

from tree_search.qwen.qwen_utils import (
    generate_initial_plan,
    modify_plan,
    evaluate_plan
)

class MetaPlanner:
    def __init__(self, generator: pipeline, question: str, file_path: str = None):
    # def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str, file_path: str = None):
        # self.model = model
        # self.tokenizer = tokenizer
        self.generator = generator
        self.question = question
        self.file_path = file_path

    def run(self):
        # First create an initial plan
        print(f"Generating initial plan for question: {self.question}")
        initial_plan = generate_initial_plan(
            generator=self.generator,
            question=self.question,
            file_path=self.file_path
        )

        # Evaluate the initial plan
        initial_score = evaluate_plan(
            generator=self.generator,
            question=self.question,
            file_path=self.file_path,
            plan=initial_plan
        )

        # Append the initial plan to the search tree
        print("Creating search tree with initial plan...")
        search_tree = SearchTree(question=self.question, initial_plan=initial_plan, initial_score=initial_score)

        print("Start expanding the search tree...")

        selected_node = search_tree.select()
        while selected_node:
            # Expand the selected node
            modifications = modify_plan(
                generator=self.generator,
                plan=selected_node.get_plan(),
                question=self.question,
                file_path=self.file_path
            )

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
            expanded_node.score = evaluate_plan(
                generator=self.generator,
                plan=plan,
                question=self.question,
                file_path=self.file_path
            )


            # Select the next node for expansion
            selected_node = search_tree.select()

        # # Print out the final search tree
        # search_tree.print_tree()

        # # Select the top k plans
        # top_k_plans = search_tree.select_top_k(top_k=3)

        # return top_k_plans
        return search_tree



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Meta Planning Runner")
    arg_parser.add_argument("--question", type=str, required=True, help="The question to answer.")
    arg_parser.add_argument("--file_path", type=str, default=None, help="Path to the file to use for context.")
    arg_parser.add_argument("--model_path_or_name", type=str, default="Qwen/Qwen2.5-32B", help="The model to use for LLM operations.")

    args = arg_parser.parse_args()

    generator = pipeline(
        "text-generation", 
        args.model_path_or_name, 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )

    # model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    runner = MetaPlanner(
        generator=generator,
        question=args.question,
        file_path=args.file_path
    )

    runner.run()
