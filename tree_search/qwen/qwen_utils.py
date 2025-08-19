from transformers import pipeline, TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
# from pydantic import BaseModel
from typing import Optional, Tuple, Callable, Any, List

from tree_search.prompts import (
    initial_planning_prompt,
    expand_prompt,
    evaluate_plan_prompt
)

from tree_search.llm_utils import (
    extract_plan,
    extract_modification,
    extract_scores
)

from tree_search.schemas import Plan, ModificationResponse, PlanScore

def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, messages: List):
    # Check if tokenizer has chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        # Use chat template
        try:
            tokenized_chat = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Chat template failed: {e}, falling back to manual formatting")
            # Fall back to manual formatting
            tokenized_chat = manual_format_messages(tokenizer, messages)
    else:
        # Manually format messages
        tokenized_chat = manual_format_messages(tokenizer, messages)
    
    # Move to same device as model
    tokenized_chat = tokenized_chat.to(model.device)
    
    # Generate response
    outputs = model.generate(
        tokenized_chat, 
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)
    return response

def manual_format_messages(tokenizer, messages):
    """Manually format messages when no chat template is available"""
    # Format messages as a conversation string
    formatted_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            formatted_text += f"System: {content}\n"
        elif role == "user":
            formatted_text += f"User: {content}\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n"
    
    # Add prompt for assistant response
    formatted_text += "Assistant:"
    
    # Tokenize the formatted text
    return tokenizer.encode(formatted_text, return_tensors="pt")

def generate_structured_response(generator: pipeline, streamer: TextStreamer, system_prompt: str, user_prompt: str, extract_func: Callable[[str], Tuple[str, Optional[Any]]]):
# def generate_structured_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str, extract_func: Callable[[str], Tuple[str, Optional[Any]]]):
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    while True:
        messages = generator(messages, max_new_tokens=1024, streamer=streamer)[0]["generated_text"]
        # print(messages[-1]["content"])
        response = messages[-1]["content"]

        # response = generate_response(model, tokenizer, messages)

        # tokenizer = generator.tokenizer
        # if hasattr(tokenizer, 'chat_template'):
        #     try:
        #         prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        #         result = generator(prompt, max_new_tokens=32768)[0]["generated_text"]
        #     except:
        #         # Fall back to string conversion
        #         prompt = str(messages)
        #         result = generator(prompt, max_new_tokens=32768)[0]["generated_text"]
        # else:
        #     prompt = str(messages)
        #     result = generator(prompt, max_new_tokens=32768)[0]["generated_text"]

        # # response = result[-1]["content"]
        # response = result[len(prompt):].strip()

        # debug
        # print(f"model response: {response}")

        extracted_result = extract_func(response)
        extract_message = extracted_result[0]
        extracted_data = extracted_result[1]
        if extract_message == "success":
            # if isinstance(extracted_data, schema):
            return extracted_data
            # return False, "Schema mismatch: The extracted data does not match the expected schema."
        else:
            messages.append(
                {"role": "user", "content": f"Error: {extract_message}. Please rephrase your response."}
            )

def generate_initial_plan(generator: pipeline, streamer: TextStreamer, question: str, file_path: str = None) -> Plan:
# def generate_initial_plan(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str, file_path: str = None) -> Plan:
    # build up user prompt

    if not file_path:
        user_prompt = f"Question: {question}"
    else:
        user_prompt = f"Question: {question}\n\nProvided File: {file_path}"

    return generate_structured_response(
        generator=generator,
        streamer=streamer,
        system_prompt=initial_planning_prompt,
        user_prompt=user_prompt,
        extract_func=extract_plan
    )

def modify_plan(generator: pipeline, streamer: TextStreamer, plan: Plan, question: str, file_path: str = None) -> ModificationResponse:
# def modify_plan(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, plan: Plan, question: str, file_path: str = None) -> ModificationResponse:
    # build up user prompt
    if file_path:
        user_prompt = f"Question: {question}\n\nProvided file: {file_path}"
    else:
        user_prompt = f"Question: {question}"

    user_prompt += f"\n\nPlan:\n{plan.model_dump_json(indent=2)}"

    return generate_structured_response(
        generator=generator,
        streamer=streamer,
        system_prompt=expand_prompt,
        user_prompt=user_prompt,
        extract_func=extract_modification
    )

def evaluate_plan(generator: pipeline, streamer: TextStreamer, plan: Plan, question: str, file_path: str = None) -> PlanScore:
# def evaluate_plan(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, plan: Plan, question: str, file_path: str = None) -> PlanScore:
    # build up user prompt
    if file_path:
        user_prompt = f"Question: {question}\n\nProvided file: {file_path}"
    else:
        user_prompt = f"Question: {question}"

    user_prompt += f"\n\nPlan:\n{plan.model_dump_json(indent=2)}"

    return generate_structured_response(
        generator=generator,
        streamer=streamer,
        system_prompt=evaluate_plan_prompt,
        user_prompt=user_prompt,
        extract_func=extract_scores
    )