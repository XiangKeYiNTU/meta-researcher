import os
import json
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import pipeline

from tree_search.qwen.meta_tree_search_runner import MetaPlanner
from plan_merger.base import PlanGraph
from agents.qwen.meta_agent import MetaAgent
from web_explorer.qwen.step_executor import StepExecutor


def get_available_devices():
    """Get list of available CUDA devices."""
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        return ["cpu"]


def initialize_worker(model_name_or_path: str, device: str):
    """Initialize worker process with model loaded on specific device."""
    global generator, qwen_client
    
    # Load model on specific device
    if device != "cpu":
        device_map = {"": device}
    else:
        device_map = "auto"
    
    generator = pipeline(
        "text-generation",
        model_name_or_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True
    )
    
    qwen_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    print(f"Worker initialized on device: {device}")


def process_single_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single GAIA task."""
    global generator, qwen_client
    
    task, task_index, total_tasks = task_data['task'], task_data['index'], task_data['total']
    
    result = {
        "task_id": task['task_id'],
        "question": task['Question'],
        "file_path": task['file_path'],
        "step_by_step_results": []
    }
    
    print(f"Worker {os.getpid()}: Processing task {task_index+1}/{total_tasks}: {task['task_id']}")
    
    try:
        question = task['Question']
        file_path = task['file_path'] if task['file_path'] != "" else None

        # Initialize planner
        plan_runner = MetaPlanner(
            generator=generator,
            question=question,
            file_path=file_path
        )

        # Run planning
        search_tree = plan_runner.run()
        top_plans = search_tree.select_top_plans()
        
        # Create plan graph
        plan_graph = PlanGraph()
        plan_graph.add_plan_list(top_plans)

        # Initialize meta agent
        meta_agent = MetaAgent(
            plan_graph=plan_graph,
            question=question,
            generator=generator
        )

        # Execute steps
        while True:
            next_step = meta_agent.generate_next_step()
            if next_step.goal == "END":
                # Finalize answer
                final_answer = meta_agent.finalize_answer()
                print(f"Worker {os.getpid()}: Final answer for {task['task_id']}: {final_answer}")
                result['final_answer'] = final_answer
                break

            print(f"Worker {os.getpid()}: Next step for {task['task_id']}: {next_step.goal}")
            finished_steps = meta_agent.plan_graph.get_current_exec_results()
            
            step_executor = StepExecutor(
                generator=generator,
                current_step=next_step,
                question=question,
                finished_steps=finished_steps,
                file_path=file_path,
                qwen_client=qwen_client
            )

            step_result = step_executor.run()
            result['step_by_step_results'].append(step_result)
            
            if "Final answer: " in step_result['result']:
                final_answer = step_result['result'].split("Final answer: ")[1].strip()
                print(f"Worker {os.getpid()}: Final answer for {task['task_id']}: {final_answer}")
                result['final_answer'] = final_answer
                break
                
            # Update graph
            step_node = meta_agent.plan_graph.exist_step(step=next_step)
            step_node.execution_result = step_result['result']

        result['status'] = 'completed'
        
    except Exception as e:
        print(f"Worker {os.getpid()}: Error processing task {task['task_id']}: {str(e)}")
        result['status'] = 'failed'
        result['error'] = str(e)
        result['final_answer'] = None

    return result


def run_parallel_gaia(args):
    """Main function to run GAIA benchmark in parallel."""
    
    # Load the GAIA benchmark
    test_set = load_dataset(
        "./dataset/GAIA/GAIA.py",
        name=f"2023_level{args.level}",
        data_dir=".",
        split=args.split,
        trust_remote_code=True
    )
    
    # Get available devices
    available_devices = get_available_devices()
    print(f"Available devices: {available_devices}")
    
    # Determine number of workers
    if args.num_workers == -1:
        num_workers = len(available_devices)
    else:
        num_workers = min(args.num_workers, len(available_devices))
    
    print(f"Using {num_workers} workers")
    
    # Prepare task data
    tasks_data = []
    for i, task in enumerate(test_set):
        if args.max_tasks > 0 and i >= args.max_tasks:
            break
        tasks_data.append({
            'task': task,
            'index': i,
            'total': min(len(test_set), args.max_tasks) if args.max_tasks > 0 else len(test_set)
        })
    
    print(f"Processing {len(tasks_data)} tasks")
    
    # Create device assignments for workers
    device_assignments = [available_devices[i % len(available_devices)] for i in range(num_workers)]
    
    # Process tasks in parallel
    results = []
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=initialize_worker,
        initargs=(args.model_name_or_path, None)  # Device will be set per process
    ) as executor:
        
        # Submit tasks with device assignments
        future_to_task = {}
        task_chunks = [tasks_data[i::num_workers] for i in range(num_workers)]
        
        for worker_id, (chunk, device) in enumerate(zip(task_chunks, device_assignments)):
            if chunk:  # Only submit if chunk is not empty
                # Reinitialize worker with specific device
                future = executor.submit(process_worker_chunk, chunk, args.model_name_or_path, device)
                future_to_task[future] = worker_id
        
        # Collect results
        for future in as_completed(future_to_task):
            worker_id = future_to_task[future]
            try:
                worker_results = future.result()
                results.extend(worker_results)
                print(f"Completed worker {worker_id}, got {len(worker_results)} results")
            except Exception as exc:
                print(f"Worker {worker_id} generated an exception: {exc}")
    
    # Sort results by task_id to maintain order
    results.sort(key=lambda x: x['task_id'])
    
    # Save results
    output_file = f"GAIA_level{args.level}_{args.split}_qwen_parallel_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    completed = sum(1 for r in results if r.get('status') == 'completed')
    failed = sum(1 for r in results if r.get('status') == 'failed')
    print(f"Summary: {completed} completed, {failed} failed out of {len(results)} total tasks")


def process_worker_chunk(chunk: List[Dict], model_name_or_path: str, device: str) -> List[Dict]:
    """Process a chunk of tasks in a single worker with specific device."""
    # Initialize worker with specific device
    initialize_worker(model_name_or_path, device)
    
    results = []
    for task_data in chunk:
        result = process_single_task(task_data)
        results.append(result)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with GAIA benchmark in parallel.")
    
    parser.add_argument("--level", type=str, default="1", help="The level of the GAIA benchmark.")
    parser.add_argument("--split", type=str, default="validation", help="The split of the GAIA benchmark.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-32B", help="Model ID or path.")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of parallel workers (-1 for auto-detect based on GPUs).")
    parser.add_argument("--max_tasks", type=int, default=-1, help="Maximum number of tasks to process (-1 for all tasks).")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of tasks per worker batch.")
    
    args = parser.parse_args()
    
    # Set multiprocessing start method (important for CUDA)
    mp.set_start_method('spawn', force=True)
    
    run_parallel_gaia(args)