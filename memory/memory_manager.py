from openai import OpenAI
from typing import List
from memory.prompts import annotation_prompt
from memory.utils import extract_experiences
from memory.schemas import AnnotatedMemory

class MemoryManager:
    def __init__(self, memory: List[AnnotatedMemory], client: OpenAI, model: str = "deepseek/deepseek-chat-v3.1:free"):
        self.memory = memory
        self.client = client
        self.model = model

    def annotate(self, question: str, step: str, actions: list, result: str, reference: str) -> AnnotatedMemory:
        # build up user prompt
        user_prompt = f"Question: {question}\n\nThe executed step: {step}\n\nExecution history:\n"
        for a in actions:
            user_prompt += f"{a['action']} {a['param']}\n"
            user_prompt += f"obtain {a['action_result']}\n\n"

        user_prompt += f"Final result: {result}\n"
        user_prompt += f"Provided reference: {reference}"

        # get annotation
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "system",
                "content": annotation_prompt
                },
                {
                "role": "user",
                "content": user_prompt
                }
            ]
        )

        experiences = extract_experiences(completion.choices[0].message.content)

        # find if step exists in memory
        for annotation in self.memory:
            if annotation.question == question:
                if annotation.step == step:
                    annotation.experiences.append(experiences)
                    return annotation
                
        annotation = AnnotatedMemory(question=question, step=step, experiences=experiences)
        return annotation
    
    