import json
import argparse
from evaluate import load
from datasets import load_dataset

class EMEvaluator:
    def __init__(self, level: int, split: str, result_path: str):
        self.level = level
        self.split = split
        self.result_path = result_path

    def load_result(self):
        """Load the evaluation results from the specified path."""
        try:
            extracted_results = []
            with open(self.result_path, 'r') as file:
                results = file.read()
            raw_results = json.loads(results)
            for result in raw_results:
                extracted_result = {}
                extracted_result['task_id'] = result['task_id']
                extracted_result['prediction'] = result['step_by_step_results'][-1]['final_answer']
                extracted_results.append(extracted_result)
            return extracted_results
        except FileNotFoundError:
            print(f"Result file not found at {self.result_path}")
            return None
        
    def load_dataset(self):
        """Load the dataset for evaluation."""
        try:
            dataset = load_dataset("../dataset/GAIA/GAIA.py", name=f"2023_level{self.level}", data_dir=".", split=self.split, trust_remote_code=True)
            extracted_dataset = []
            for item in dataset:
                extracted_item = {}
                extracted_item['task_id'] = item['task_id']
                extracted_item['gt'] = item['Final answer']
                extracted_dataset.append(extracted_item)
            return extracted_dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def evaluate(self):
        """Evaluate the results against the ground truth."""
        predictions = self.load_result()
        dataset = self.load_dataset()

        exact_match_metric = load("exact_match")

        if predictions is None or dataset is None:
            return None
        
        # Prepare the evaluation data
        prediction_list =[]
        gt_list = []
        for prediction in predictions:
            task_id = prediction['task_id']
            pred = prediction['prediction']
            prediction_list.append(pred)
            gt = next((item['gt'] for item in dataset if item['task_id'] == task_id), None)
            if gt is not None:
                gt_list.append(gt)
            else:
                gt_list.append("")
        
        em_result = exact_match_metric.compute(predictions=prediction_list, references=gt_list)

        return round(em_result['exact_match'], 4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the GAIA benchmark results.")
    parser.add_argument("--level", type=int, default=1, help="The level of the GAIA benchmark.")
    parser.add_argument("--split", type=str, default="validation", help="The split of the GAIA benchmark.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result JSON file.")

    args = parser.parse_args()

    evaluator = EMEvaluator(level=args.level, split=args.split, result_path=args.result_path)
    em_score = evaluator.evaluate()

    if em_score is not None:
        print(f"Exact Match Score for GAIA Level {args.level}, Split {args.split}: {em_score}")
    else:
        print("Evaluation failed due to missing data.")