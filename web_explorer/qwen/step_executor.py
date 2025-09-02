from tree_search.schemas import Step
from typing import List, Tuple
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import torch

from transformers import pipeline, TextStreamer

from web_explorer.prompts import system_prompt
from web_explorer.utils import extract_action, truncate_markdown, summarize_web_content_by_qwen
from web_explorer.search_api import get_text_search_results
from web_explorer.visit_api import visit

from document_tools.document_parser import DocumentParser

class StepExecutor:
    def __init__(self, generator: pipeline, streamer: TextStreamer, current_step: Step, qwen_client: OpenAI, 
                 finished_steps: List[Tuple[Step, str]] = None, file_path: str = None, max_context_tokens: int = 16000):
        self.generator = generator
        self.streamer = streamer
        self.finished_steps = finished_steps or []
        self.current_step = current_step
        self.file_path = file_path
        self.qwen_client = qwen_client
        self.max_context_tokens = max_context_tokens
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def truncate_old_messages(self, messages: list, preserve_count: int = 3) -> list:
        """Keep system message, initial user prompt, and last N exchanges"""
        if len(messages) <= preserve_count + 2:  # +2 for system and initial user
            return messages
        
        # Always keep: system message, initial user message, last N exchanges
        system_msg = messages[0]
        initial_user = messages[1]
        recent_messages = messages[-(preserve_count * 2):]  # Last N exchanges (user + assistant pairs)
        
        return [system_msg, initial_user] + recent_messages
    
    def create_context_summary(self, actions: list, extracted_info: list) -> str:
        """Create a compact summary of previous actions"""
        if not actions:
            return ""
        
        summary = "Previous actions summary:\n"
        
        # Group actions by type
        searches = [a['param'] for a in actions if a["action"] == "search"]
        visits = [a['param'] for a in actions if a["action"] == "visit"]
        extracts = [a['param'] for a in actions if a["action"] == "extract"]
        
        if searches:
            keywords = ' '.join(searches)
            summary += f"- Performed {len(searches)} searches with keywords: {keywords}\n"
        if visits:
            url = '\n'.join(visits)
            summary += f"- Visited {len(visits)} websites:\n{url}\n"
        if extracts:
            summary += f"- Extracted {len(extracts)} pieces of information\n"
        
        # Include extracted info (most important)
        if extracted_info:
            summary += "\nExtracted information:\n"
            for info in extracted_info:  # Only last 5 extractions
                summary += f"- {info}\n"
        
        return summary
    
    def run(self):
        # Initialize previous steps context
        if not self.finished_steps:
            previous_steps = "You are executing the first step."
        else:
            # Summarize previous steps instead of including full details
            previous_steps = f"Previous steps completed: {len(self.finished_steps)}\n"
            # Only include the last 2 completed steps in detail
            for step, answer in self.finished_steps:
                previous_steps += f"Step: {step.goal}\nAnswer: {answer}\n\n"

        user_prompt = f"Previous steps:\n{previous_steps}Current step: {self.current_step.goal}\n\nInstructions: {self.current_step.instructions}"

        # Parse file content (truncate if too long)
        if self.file_path:
            parser = DocumentParser()
            file_content = parser.parse_file(file_path=self.file_path)
            if self.estimate_tokens(file_content) > 5000:
                file_content = file_content[:20000] + "\n... [File truncated for context limits]"
            user_prompt += f"\n\nFile content:\n{file_content}"

        # Initialize execution state
        extracted_info = []
        search_count = 0
        visit_count = 0
        actions = []
        search_cache = {}
        visit_cache = {}
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        iteration_count = 0
        max_iterations = 20  # Prevent infinite loops
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            # Context management: truncate messages if getting too long
            total_context = sum(self.estimate_tokens(str(msg)) for msg in messages)
            if total_context > self.max_context_tokens:
                messages = self.truncate_old_messages(messages, preserve_count=2)
                
                # Add context summary
                context_summary = self.create_context_summary(actions, extracted_info)
                if context_summary:
                    messages.insert(-1, {"role": "user", "content": context_summary})
            
            # Generate response
            print(f"Current message stream length: {len(messages)}")
            result = self.generator(messages, max_new_tokens=1024, streamer=self.streamer)  # Reduced token count
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                # Handle pipeline output format
                response = result[0]["generated_text"]
                if isinstance(response, list):
                    response = response[-1]["content"]
                else:
                    # Extract new content only
                    current_length = sum(len(str(msg.get("content", ""))) for msg in messages)
                    response = response[current_length:].strip()
            else:
                response = str(result)
            
            # print(f"Model response (iteration {iteration_count}): {response[:200]}...")
            
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response})
            
            # Execute action
            action = extract_action(response)
            if not action:
                messages.append({
                    "role": "user", 
                    "content": "Please specify an action using: <search>, <visit>, <extract>, or <answer>"
                })
                continue
                
            action_step = {"action": action[0], "param": action[1] if len(action) > 1 else None}
            
            if action[0] == "search":
                if self._handle_search_action(action, search_count, search_cache, messages, actions, action_step):
                    search_count += 1
                    if search_count >= 5:
                        break
                        
            elif action[0] == "visit":
                if self._handle_visit_action(action, visit_count, visit_cache, messages, actions, action_step):
                    visit_count += 1
                    if visit_count >= 10:
                        break
                        
            elif action[0] == "extract":
                self._handle_extract_action(action, extracted_info, messages, actions, action_step)
                
            elif action[0] == "summary":
                return self._create_step_results(action, extracted_info, search_count, actions)
            
            # elif action[0] == "skip":
            #     return self._create_step_results(action, extracted_info, search_count, actions)
                
            # elif action[0] == "finalize":
            #     return self._create_final_results(action, extracted_info, search_count, actions)
                
            else:
                messages.append({
                    "role": "user",
                    "content": "Invalid action. Use: <search>, <visit>, <extract>, or terminating step using <answer>"
                })
        
        # If max iterations reached, force final summary from model
        return self._force_final_summary(messages, extracted_info, search_count, actions)
    
    def _handle_search_action(self, action, search_count, search_cache, messages, actions, action_step):
        if search_count >= 10:
            messages.append({
                "role": "user",
                "content": "Search quota reached. Please provide your answer after <answer>."
            })
            action_step["action_result"] = "search quota reached"
            actions.append(action_step)
            return False
        
        query = action[1]
        if query in search_cache:
            search_results = search_cache[query]
            result_string = json.dumps(search_results, indent=2)
            user_prompt = "CACHED SEARCH:\n```search_results\n" + result_string + "\n```"
        else:
            search_results = get_text_search_results(query)
            search_cache[query] = search_results
            result_string = json.dumps(search_results, indent=2)  # Truncate long results
            user_prompt = "```search_results\n" + result_string + "\n```"
        
        messages.append({"role": "user", "content": user_prompt})
        action_step["action_result"] = result_string
        actions.append(action_step)
        return True
    
    def _handle_visit_action(self, action, visit_count, visit_cache, messages, actions, action_step):
        if visit_count >= 50:
            messages.append({
                "role": "user",
                "content": "Visit quota reached. Please provide your answer using <answer>."
            })
            action_step["action_result"] = "visit quota reached"
            actions.append(action_step)
            return False
        
        url = action[1]
        topic = action[2]
        if not topic:
                topic = self.current_step.goal
        if url in visit_cache:
            web_content = visit_cache[url]
            web_summary = summarize_web_content_by_qwen(topic, web_content, self.qwen_client)
            user_prompt = "CACHED VISIT:\n"
            user_prompt += f"Website summary:\n```web_content\n{str(web_summary)}\n```"
        else:
            raw_content = visit(url)
            short_content = truncate_markdown(raw_content)  # Reduced token limit
            # pre define topic
            # topic = self.current_step.goal
            web_summary = summarize_web_content_by_qwen(topic, short_content, self.qwen_client)
            user_prompt = f"Website summary:\n```web_content\n{str(web_summary)}\n```"  # Truncate summary
            visit_cache[url] = short_content
        
        # short_content = truncate_markdown(raw_content, max_tokens=8000)  # Reduced token limit
        # web_summary = summarize_web_content_by_qwen(self.current_step.goal, short_content, self.qwen_client)
        # user_prompt += f"Website summary:\n```web_content\n{str(web_summary)[:1500]}\n```"  # Truncate summary
        
        messages.append({"role": "user", "content": user_prompt})
        action_step["action_result"] = str(web_summary)
        actions.append(action_step)
        return True
    
    def _handle_extract_action(self, action, extracted_info, messages, actions, action_step):
        extracted_info.append(action[1])
        # Only show recent extractions to save context
        recent_extractions = extracted_info[-5:] if len(extracted_info) > 5 else extracted_info
        user_message = f"Recent extractions ({len(recent_extractions)}/{len(extracted_info)}):\n"
        for info in recent_extractions:
            user_message += f"- {info}\n"

        user_message += f"Decide if the info is enough to answer. If enough, use <answer> to give answer of the step. If not enough, decide the next <search> or <visit> action."
        
        messages.append({"role": "user", "content": user_message})
        action_step["action_result"] = extracted_info[-1]  # Only store the latest extraction
        actions.append(action_step)
    
    def _force_final_summary(self, messages, extracted_info, search_count, actions):
        """Force the model to provide a final summary when max iterations reached"""
        
        # Create a comprehensive prompt for final summary
        summary_prompt = f"""
MAXIMUM ITERATIONS REACHED - FINAL SUMMARY REQUIRED

Current step goal: {self.current_step.goal}

Extracted information so far:
"""
        
        if extracted_info:
            for i, info in enumerate(extracted_info, 1):
                summary_prompt += f"{i}. {info}\n"
        else:
            summary_prompt += "No information extracted yet.\n"
        
        summary_prompt += f"""
Actions performed: {len(actions)} total actions
- Searches: {search_count}
- Visits: {len([a for a in actions if a["action"] == "visit"])}
- Extractions: {len([a for a in actions if a["action"] == "extract"])}

Please provide a comprehensive summary of what you have found relevant to the current step goal using <answer>.
"""
        
        # Add the summary prompt
        messages.append({"role": "user", "content": summary_prompt})
        
        # Get model's final response
        try:
            result = self.generator(messages, max_new_tokens=2048)
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                response = result[0]["generated_text"]
                if isinstance(response, list):
                    response = response[-1]["content"]
                else:
                    # Extract new content only
                    current_length = sum(len(str(msg.get("content", ""))) for msg in messages)
                    response = response[current_length:].strip()
            else:
                response = str(result)
            
            print(f"Forced final summary response: {response}")
            
            # Try to extract summary or finalize action from response
            action = extract_action(response)
            
            if action and action[0] == "summary":
                # Use the extracted action
                return self._create_step_results(action, extracted_info, search_count, actions)
            else:
                # If no proper action format, use the entire response as summary
                fallback_action = ["summary", response]
                return self._create_step_results(fallback_action, extracted_info, search_count, actions)
                
        except Exception as e:
            print(f"Error during forced summary generation: {e}")
            # Fallback to a basic summary with extracted info
            fallback_summary = f"Maximum iterations reached. Extracted information: {'; '.join(extracted_info) if extracted_info else 'None'}"
            fallback_action = ["summary", fallback_summary]
            return self._create_step_results(fallback_action, extracted_info, search_count, actions)
    
    def _create_step_results(self, action, extracted_info, search_count, actions):
        reference = action[2]
        step_results = {
            "goal": self.current_step.goal,
            "result": action[1],
            "reference": reference if reference else "No reference provided",
            "found_relevant_info": extracted_info,
            "search_count": search_count,
            "visit_count": len([a for a in actions if a["action"] == "visit"]),
            "actions": actions
        }
        # self.finished_steps.append(step_results)
        return step_results
    
    # def _create_final_results(self, action, extracted_info, search_count, actions):
    #     return {
    #         "goal": self.current_step.goal,
    #         "result": f"Final answer: {action[1]}",
    #         "found_relevant_info": extracted_info,
    #         "search_count": search_count,
    #         "visit_count": len([a for a in actions if a["action"] == "visit"]),
    #         "total_actions": len(actions)
    #     }


if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser(description="Meta Planning Runner")
    # arg_parser.add_argument("--question", type=str, required=True, help="The question to answer.")
    # arg_parser.add_argument("--file_path", type=str, default=None, help="Path to the file to use for context.")
    # arg_parser.add_argument("--model_path_or_name", type=str, default="Qwen/Qwen2.5-32B", help="The model to use for LLM operations.")

    # args = arg_parser.parse_args()

    generator = pipeline(
        "text-generation", 
        "Qwen/Qwen2.5-32B", 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )

    # Get the path to the parent folder
    parent_env_path = Path(__file__).resolve().parents[2] / ".env"
    # Load the .env file from the parent folder
    load_dotenv(dotenv_path=parent_env_path)

    qwen_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    step_executor = StepExecutor(
        generator=generator,
        current_step=Step(goal="Search for the tracklist of the VNV Nation album Futureperfect", instructions="Search for the album \"Futureperfect\" by VNV Nation and look through the tracklist"),
        question="One of the songs on the VNV Nation album Futureperfect has a non-English title. The title references another piece of music. Who composed it? Answer using the format First name Last name.",
        qwen_client=qwen_client
    )

    results = step_executor.run()

    print(f"execution result: {results['result']}")