import os
from openai import OpenAI

import argparse

from transformers import pipeline, TextStreamer

from tree_search.qwen.meta_tree_search_runner import MetaPlanner
from plan_merger.base import PlanGraph
from agents.qwen.meta_agent import MetaAgent
from web_explorer.qwen.step_executor import StepExecutor

from datasets import load_dataset

import json

import traceback
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_tasks_with_error_handling(test_set: List[Dict], meta_generator, executor_generator, meta_streamer, executor_streamer, qwen_client, max_retries: int = 2):
    """
    Process tasks with comprehensive error handling.
    
    Args:
        test_set: List of tasks to process
        generator: The model generator
        qwen_client: Qwen client for summarization
        max_retries: Number of retries per task before giving up
    
    Returns:
        List of results including successful and failed task outcomes
    """
    result_json = []
    successful_tasks = 0
    failed_tasks = 0
    
    for i, task in enumerate(test_set):
        task_id = task.get('Record ID', f'task_{i}')
        logger.info(f"Starting task {i+1}/{len(test_set)}: {task_id}")
        
        result = process_single_task(task, meta_generator, executor_generator, meta_streamer, executor_streamer, qwen_client, i+1, len(test_set), max_retries)
        result_json.append(result)
        
        if result.get('status') == 'success':
            successful_tasks += 1
        else:
            failed_tasks += 1
        # else:
        #     logger.info(f"Skipping task {task_id} - has file_path")
        #     result = {
        #         "task_id": task_id,
        #         "question": task.get('Question', ''),
        #         "status": "skipped",
        #         "error": "Task has file_path - skipped per condition",
        #         "step_by_step_results": []
        #     }
        #     result_json.append(result)
    
    logger.info(f"Task processing completed. Successful: {successful_tasks}, Failed: {failed_tasks}, Total: {len(test_set)}")
    return result_json

def process_single_task(task: Dict, meta_generator, executor_generator, meta_streamer, executor_streamer, qwen_client, task_num: int, total_tasks: int, max_retries: int = 2) -> Dict[str, Any]:
    """
    Process a single task with error handling and retries.
    
    Returns:
        Dict containing task result with status information
    """
    task_id = task.get('Record ID', f'task_{task_num}')
    question = task.get('Question', '')
    domain = task.get('High-level domain', '')
    
    # Initialize result structure
    result = {
        "task_id": task_id,
        "question": question,
        "domain": domain,
        "step_by_step_results": [],
        "status": "failed",  # Will be updated to "success" if completed
        "error": None,
        "retry_count": 0
    }
    
    for retry_attempt in range(max_retries + 1):
        try:
            logger.info(f"Processing task {task_num}/{total_tasks}: {task_id} (attempt {retry_attempt + 1})")
            
            # Reset step results for retry attempts
            if retry_attempt > 0:
                result["step_by_step_results"] = []
                result["retry_count"] = retry_attempt
                logger.warning(f"Retrying task {task_id} - attempt {retry_attempt + 1}/{max_retries + 1}")
            
            file_path = None
            
            # Step 1: Initialize MetaPlanner
            try:
                plan_runner = MetaPlanner(generator=meta_generator, streamer=meta_streamer, question=question, file_path=file_path)
            except Exception as e:
                raise TaskProcessingError(f"Failed to initialize MetaPlanner: {str(e)}", "meta_planner_init")
            
            # Step 2: Generate search tree and plans
            try:
                search_tree = plan_runner.run()
                top_plans = search_tree.select_top_plans()
            except Exception as e:
                raise TaskProcessingError(f"Failed to generate plans: {str(e)}", "plan_generation")
            
            # Step 3: Initialize plan graph and meta agent
            try:
                plan_graph = PlanGraph()
                plan_graph.add_plan_list(top_plans)
                meta_agent = MetaAgent(plan_graph=plan_graph, question=question, generator=meta_generator, streamer=meta_streamer)
            except Exception as e:
                raise TaskProcessingError(f"Failed to initialize MetaAgent: {str(e)}", "meta_agent_init")
            
            # Step 4: Execute steps
            max_execution_steps = 50  # Prevent infinite loops
            execution_steps = 0
            
            while execution_steps < max_execution_steps:
                try:
                    execution_steps += 1
                    
                    # Generate next step
                    try:
                        next_step = meta_agent.generate_next_step()
                        # Meta agent chooses to skip
                        if not next_step:
                            continue
                    except Exception as e:
                        logger.error(f"Error generating next step for task {task_id}: {str(e)}")
                        # Try to finalize with current progress
                        break
                    
                    # Check if execution should end
                    if next_step.goal == "END":
                        try:
                            final_answer = meta_agent.finalize_answer()
                            logger.info(f"Task {task_id} completed with final answer: {final_answer[:100]}...")
                            result['final_answer'] = final_answer
                            result['status'] = 'success'
                            return result
                        except Exception as e:
                            raise TaskProcessingError(f"Failed to finalize answer: {str(e)}", "finalize_answer")
                    
                    logger.info(f"Task {task_id} - executing step {execution_steps}: {next_step.goal}")
                    
                    # Execute current step
                    try:
                        finished_steps = meta_agent.plan_graph.get_current_exec_results()
                        step_executor = StepExecutor(
                            generator=executor_generator,
                            streamer=executor_streamer,
                            current_step=next_step,
                            finished_steps=finished_steps,
                            file_path=file_path,
                            qwen_client=qwen_client
                        )
                        
                        step_result = step_executor.run()
                        result['step_by_step_results'].append(step_result)
                        
                        # Check if this step provided a final answer
                        if step_result.get('result', '') and "Final answer: " in step_result['result']:
                            final_answer = step_result['result'].split("Final answer: ")[1].strip()
                            logger.info(f"Task {task_id} completed via step result: {final_answer[:100]}...")
                            result['final_answer'] = final_answer
                            result['status'] = 'success'
                            return result
                        
                        # Update graph with step result
                        try:
                            step_node = meta_agent.plan_graph.exist_step(step=next_step)
                            if step_node:
                                step_node.execution_result = step_result.get('result', '')
                        except Exception as e:
                            logger.warning(f"Failed to update plan graph for task {task_id}: {str(e)}")
                            # Continue execution even if graph update fails
                        
                    except Exception as e:
                        logger.error(f"Error executing step '{next_step.goal}' for task {task_id}: {str(e)}")
                        # Log the error but continue to next step
                        error_step_result = {
                            "goal": next_step.goal,
                            "result": f"Step failed with error: {str(e)}",
                            "error": str(e),
                            "status": "failed"
                        }
                        result['step_by_step_results'].append(error_step_result)
                        continue
                
                except Exception as e:
                    logger.error(f"Unexpected error in execution loop for task {task_id}: {str(e)}")
                    break
            
            # If we exit the while loop without completion
            if execution_steps >= max_execution_steps:
                raise TaskProcessingError(f"Maximum execution steps ({max_execution_steps}) reached", "max_steps_exceeded")
            else:
                raise TaskProcessingError("Execution loop ended without completion", "incomplete_execution")
            
        except TaskProcessingError as e:
            logger.error(f"Task processing error for {task_id} (attempt {retry_attempt + 1}): {e.message} (stage: {e.stage})")
            result['error'] = f"{e.stage}: {e.message}"
            
            if retry_attempt < max_retries:
                continue  # Retry
            else:
                result['status'] = 'failed'
                return result
                
        except Exception as e:
            logger.error(f"Unexpected error processing task {task_id} (attempt {retry_attempt + 1}): {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            result['error'] = f"Unexpected error: {str(e)}"
            
            if retry_attempt < max_retries:
                continue  # Retry
            else:
                result['status'] = 'failed'
                return result
    
    # Should not reach here, but just in case
    result['status'] = 'failed'
    result['error'] = 'All retry attempts exhausted'
    return result

class TaskProcessingError(Exception):
    """Custom exception for task processing errors with stage information"""
    def __init__(self, message: str, stage: str):
        self.message = message
        self.stage = stage
        super().__init__(f"{stage}: {message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta Planning Runner with GPQA-Diamond benchmark.")

    # parser.add_argument("--level", type=str, default="1", help="The level of the GAIA benchmark.")
    # parser.add_argument("--split", type=str, default="validation", help="The split of the GAIA benchmark.")
    parser.add_argument("--meta_model_name_or_path", type=str, default="Qwen/Qwen2.5-32B", help="Meta model ID or path.")
    parser.add_argument("--executor_model_name_or_path", type=str, default="Qwen/Qwen2.5-32B", help="Executor model ID or path.")

    args = parser.parse_args()

    meta_generator = pipeline(
        "text-generation", 
        args.meta_model_name_or_path, 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )

    executor_generator = pipeline(
        "text-generation", 
        args.executor_model_name_or_path, 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )

    meta_streamer = TextStreamer(meta_generator.tokenizer, skip_prompt=True, skip_special_tokens=True)

    executor_streamer = TextStreamer(executor_generator.tokenizer, skip_prompt=True, skip_special_tokens=True)

    qwen_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )

    # load the GPQA-Diamond benchmark
    test_set = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    # result_json = process_tasks_with_error_handling(test_set, meta_generator, executor_generator, executor_streamer=executor_streamer, qwen_client, max_retries=2)
    result_json = process_tasks_with_error_handling(test_set=test_set,
                                                    meta_generator=meta_generator,
                                                    executor_generator=executor_generator,
                                                    meta_streamer=meta_streamer,
                                                    executor_streamer=executor_streamer,
                                                    qwen_client=qwen_client)

    with open(f"GPQA_qwen_results.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4)

    successful = sum(1 for r in result_json if r.get('status') == 'success')
    failed = sum(1 for r in result_json if r.get('status') == 'failed')
    skipped = sum(1 for r in result_json if r.get('status') == 'skipped')
    
    print(f"\nTask Processing Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(result_json)}")