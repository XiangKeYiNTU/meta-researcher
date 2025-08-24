import os
import json
from openai import OpenAI
from llm_utils import generate_plan
from schemas import Dataset, Data

class MusiqueCollector:
    def __init__(self, qa_set_path: str, max_question_num: int, client: OpenAI, current_data_path: str = None, model: str = "deepseek/deepseek-r1-0528:free"):
        self.qa_set_path = qa_set_path
        self.max_question_num = max_question_num
        self.current_data_path = current_data_path
        self.client = client

    def load_qa_set(self):
        with open(self.qa_set_path, 'r') as f:
            raw = f.read()

        raw_qa_set = json.loads(raw)
        qa_set = []
        for qa in raw_qa_set:
            processed_qa = {"question": qa['question'], "answer": qa['answer'], "answerable": qa['answerable']}
            qa_set.append(processed_qa)

        return qa_set

    def collect(self) -> Dataset:
        if self.current_data_path and os.path.exists(self.current_data_path):
            print("loading current data...")
            with open(self.current_data_path, 'r') as f:
                raw = f.read()

            cur_dataset_obj = json.loads(raw)
            cur_dataset = Dataset.model_validate(cur_dataset_obj)
            print("loaded")
        else:
            print("No current data provided, create new dataset...")
            cur_dataset = Dataset(data=[])

        if len(cur_dataset.data) >= self.max_question_num:
            return cur_dataset
        else:
            print("Loading QA set...")
            qa_set = self.load_qa_set()
            i = cur_dataset.cur_index
            
            while len(cur_dataset.data) < self.max_question_num and i < len(qa_set):
                # load question and answer, and generate plan
                question = qa_set[i]['question']
                answer = qa_set[i]['answer']
                if qa_set[i]['answerable']:
                    print(f"Collecting question {i}...")
                    plan1 = generate_plan(client=self.client, question=question, answer=answer)
                    # plan2 = generate_plan(client=self.client, question=question, answer=answer, model="qwen/qwen3-235b-a22b:free")
                    data_point = Data(question=question, gt_answer=answer, plans=[plan1])
                    cur_dataset.data.append(data_point)
                    cur_dataset.cur_index = i + 1  # Update to next index
                    print("Done collecting\n")
                i += 1

                # Save after each data point is added (optional: for crash recovery)
                self._save_dataset(cur_dataset)

            # Final save to ensure all data is persisted
            self._save_dataset(cur_dataset)
            print("Dataset saved")
            return cur_dataset
        
    def _save_dataset(self, dataset: Dataset):
        if not self.current_data_path:
            data_path = "~/scratch/meta-researcher/verl/sapo/dataset/processed/musique_data.json"
        else:
            data_path = self.current_data_path
        
        # Convert dataset to dict for JSON serialization
        dataset_dict = dataset.model_dump()
        
        with open(data_path, 'w') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)