import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from evaluation.openai.gaia_evaluator import GAIAEvaluator

if __name__ == "__main__":
    load_dotenv()


    parser = argparse.ArgumentParser(description="Evaluate the GAIA benchmark results.")
    parser.add_argument("--level", type=int, default=1, help="The level of the GAIA benchmark.")
    parser.add_argument("--split", type=str, default="validation", help="The split of the GAIA benchmark.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result JSON file.")

    args = parser.parse_args()

    client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

    evaluator = GAIAEvaluator(level=args.level, split=args.split, client=client)  # Replace None with your OpenAI client instance
    flags = evaluator.evaluate_complete_result(args.result_path)
    accuracy = sum(flags) / len(flags) if flags else 0.0
    print(f"Evaluation completed. Accuracy: {accuracy:.4f}")