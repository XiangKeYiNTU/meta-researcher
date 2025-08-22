import json
from openai import OpenAI
from llm_utils import generate_plan
from schemas import Dataset, Data

class HotpotPlanner:
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
            processed_qa = {"question": qa['question'], "answer": qa['answer']}
            qa_set.append(processed_qa)

        return qa_set

    def collect(self) -> Dataset:
        with open(self.current_data_path, 'r') as f:
            raw = f.read()

        cur_dataset_obj = json.loads(raw)

        if len(cur_dataset_obj) >= self.max_question_num:
            return Dataset.model_validate(cur_dataset_obj)
        else:
            qa_set = self.load_qa_set()
            cur_dataset = Dataset.model_validate(cur_dataset_obj)
            i = cur_dataset.cur_index
            while len(cur_dataset.data) < self.max_question_num and i < len(qa_set):
                # load question and answer, and generate plan
                question = qa_set[i]['question']
                answer = qa_set[i]['answer']
                plan1 = generate_plan(client=self.client, question=question, answer=answer)
                plan2 = generate_plan(client=self.client, question=question, answer=answer, model="qwen/qwen3-235b-a22b:free")
                data_point = Data(question=question, gt_answer=answer, plans=[plan1, plan2])
                cur_dataset.data.append(data_point)
                cur_dataset.cur_index = i
                i += 1

            return cur_dataset


        

        