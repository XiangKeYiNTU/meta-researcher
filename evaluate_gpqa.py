import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from evaluation.openai.gpqa_evaluator import GPQAEvaluator

if __name__ == "__main__":
    load_dotenv()


    parser = argparse.ArgumentParser(description="Evaluate the GPQA benchmark results.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result JSON file.")

    args = parser.parse_args()

    client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

    evaluator = GPQAEvaluator(client=client)  # Replace None with your OpenAI client instance
    result = evaluator.evaluate_complete_result(args.result_path)
    domain_result = {}
    for r in result:
        domain = r['domain']
        if domain not in domain_result:
            domain_result[domain] = {'correct': 0, 'total': 0}
        if r['flag']:
            domain_result[domain]['correct'] += 1
        domain_result[domain]['total'] += 1

    for domain, counts in domain_result.items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        print(f"Domain: {domain}, Accuracy: {accuracy:.4f} ({counts['correct']}/{counts['total']})")

    overall_accuracy = sum(counts['correct'] for counts in domain_result.values()) / sum(counts['total'] for counts in domain_result.values())
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
