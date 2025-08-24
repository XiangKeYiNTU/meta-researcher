import os
# import json
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from hotpot_collector import HotpotCollector

if __name__ == "__main__":
    # with open('hotpot_dev_distractor_v1.json', 'r') as f:
    #     raw = f.read()

    # ds = json.loads(raw)
    # print(f"number of question: {len(ds)}\n{ds[0].keys()}")

    # ds = []
    # with open('musique_ans_v1.0_train.jsonl', 'r') as f:
    #     for line in f:
    #         line = line.strip()
    #         if line:  # skip empty lines
    #             ds.append(json.loads(line))

    # print(f"number of questions: {len(ds)}")
    # print(ds[0].keys())
    # print(ds[0]['answerable'])

    # Get the path to the parent folder
    parent_env_path = Path(__file__).resolve().parents[2] / ".env"
    # Load the .env file from the parent folder
    load_dotenv(dotenv_path=parent_env_path)

    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )



    hotpotCollector = HotpotCollector(qa_set_path='~/scratch/meta-researcher/verl/sapo/dataset/raw/hotpot_dev_distractor_v1.json', max_question_num=10, client=client)
    hotpot_dataset = hotpotCollector.collect()