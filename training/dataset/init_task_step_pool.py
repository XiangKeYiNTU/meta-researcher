import json

if __name__ == "__main__":
    # with open('musique_full_v1.0_train.jsonl', 'r') as f:
    #     raw = f.read()

    # ds = json.loads(raw)
    # print(f"number of question: {len(ds)}\n{ds[0].keys()}")

    ds = []
    with open('musique_full_v1.0_train.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                ds.append(json.loads(line))

    print(f"number of questions: {len(ds)}")
    print(ds[0].keys())