# Meta-thinker (enhancing deep research agents with meta-planning and transferable memory)

## Features

🤖 A meta-agent that provides detailed steps and instructions for a specified task using transferable memory through tree-search planning and plan merging.

![Tree Planning](./assets/tree_planning.png)
![Plan Merging](./assets/meta_graph.png)

🔎 A web exploring agent that is able to conduct web search and visit websites.

![Meta Execution](./assets/meta_completed.png)

## Reproduction

🔌Currently supported providers:
- OpenAI
- Qwen series (hosted by Transformers)

For reproducing results on GAIA benchmark:

1. Clone the repository and enter the directory:

```shell
git clone https://github.com/XiangKeYiNTU/meta-researcher.git
cd meta-researcher
```

2. Download the GAIA benchmark from [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA/tree/main), make a `dataset` folder under the repo folder, and move the downloaded dataset into it.

When finished, check if the benchmark is downloaded and moved correctly:

```shell
ls dataset/
> GAIA

ls dataset/GAIA/
> 2023  GAIA.py  README.md
```

3. Create conda environment and install required packages

```shell
conda create -n meta-researcher
conda activate meta-researcher
pip install -r requirements.txt
```

4. Copy `.env.example`, rename it as `.env`, and add your API keys

```shell
cp .env.example .env
```

The API keys needed:

```shell
OPENAI_API_KEY=
SERP_API_KEY=
JINA_API_KEY=
OPENROUTER_API_KEY=
HF_TOKEN=
```

5. Execute

For OpenAI hosted models, run:
```
python run_gaia_openai.py --level "1" \
    --split "validation" \
    --planner_model "gpt-4o-mini" \
    --planner_model "gpt-4o-mini" \
    --executor_model "gpt-4o-mini"
```

The default tested split is "validation" and the level is 1 if not specified. The default model backbone for all agent roles is "gpt-4o-mini" if not specified.

For Qwen models, run:
```
python run_gaia_qwen.py --level 1 \
    --split validation \
    --meta_model_name_or_path Qwen/Qwen2.5-32B \
    --executor_model_name_or_path Qwen/Qwen2.5-32B \
```

For running on multiple devices in parallel:
```
# Auto-detect devices and use all available GPUs
python run_gaia_qwen_multi.py --level 1 \
    --split validation \
    --model_name_or_path Qwen/Qwen2.5-32B

# Use specific number of workers
python run_gaia_qwen_multi.py --level 1 \
--split validation \
--model_name_or_path Qwen/Qwen2.5-32B \
--num_workers 4

# Process only first 10 tasks for testing
python run_gaia_qwen_multi.py --level 1 \
--split validation \
--model_name_or_path Qwen/Qwen2.5-32B \
--max_tasks 10 \
```

The default tested split is "validation" and the level is 1 if not specified. The default model ID is "Qwen/Qwen2.5-32B" if not specified.

The execution results will be saved into `GAIA_level1_validation_results.json`.

6. Evaluate the results

First download the repo from [exact_match](https://huggingface.co/spaces/evaluate-metric/exact_match/tree/main) and run `evaluate_gaia.py` to get the results on EM (Exact Metric), specify the level, split, and the result JSON path

```shell
python evaluate_gaia.py --level 1 --split "validation" --result_path "./GAIA_level1_validation_results.json"
```

**BONUS**: you can also run inference on a single question using the following command:

```shell
python run_single_question_openai.py "Your question here" [--file_path "The path of the given file"] (Optional)

python run_single_question_qwen.py "Your question here" [--file_path "The path of the given file"] (Optional)
```

## More to come

**Sample memory and memory retrieving module**

**Executor agent RL training scripts with multi-turn tool use**
