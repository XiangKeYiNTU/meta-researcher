import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

from openai import OpenAI
from typing import List, Tuple
from memory.prompts import annotation_prompt
from memory.utils import extract_experiences
from memory.schemas import AnnotatedMemory

class MemoryManager:
    def __init__(self, memory: List[AnnotatedMemory], client: OpenAI, model: str = "deepseek/deepseek-chat-v3.1:free"):
        self.memory = memory
        self.client = client
        self.model = model

    def add(self, question: str, step: str, actions: list, result: str, reference: str) -> AnnotatedMemory:
        # build up user prompt
        user_prompt = f"Question: {question}\n\nThe executed step: {step}\n\nExecution history:\n"
        for a in actions:
            user_prompt += f"{a['action']} {a['param']}\n"
            user_prompt += f"obtain {a['action_result']}\n\n"

        user_prompt += f"Final result: {result}\n"
        user_prompt += f"Provided reference: {reference}"

        # get annotation
        print(f"Summarizing experiences for execution of the step: {step}...")
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

        exp_str = '\n'.join(experiences)
        print(f"Summarized experiences:\n{exp_str}")

        # find if step exists in memory
        question_emb = None
        for annotation in self.memory:
            if annotation.question == question:
                question_emb = annotation.question_emb
                if annotation.step == step:
                    # step_emb = annotation.step_emb
                    annotation.experiences.append(experiences)
                    print("Experiences added in memory")
                    return annotation

        step_emb = model.encode(step, normalize_embeddings=True)
        # find if question exists in the memory
        if not question_emb:
            question_emb = model.encode(question, normalize_embeddings=True)

        annotation = AnnotatedMemory(question=question, step=step, experiences=experiences, question_emb=question_emb, step_emb=step_emb)
        self.memory.append(annotation)
        print("Experiences added in memory")
        return annotation

    def retrieve(
        self,
        question: str,
        step: str,
        alpha: float = 0.7,
        topk: int = 3
    ) -> List[Tuple[AnnotatedMemory, float]]:
        """
        Retrieve entries of memories based on weighted similarities of question and step.
        Returns a list of tuples: (AnnotatedMemory, similarity_score)
        """

        if not self.memory:
            return []

        # Compute embeddings for query
        # Assuming self.model has an encode method for embeddings
        question_emb = self.model.encode(question, normalize_embeddings=True)
        step_emb = self.model.encode(step, normalize_embeddings=True)

        results = []

        for annotation in self.memory:
            # Compute cosine similarity separately
            sim_q = cosine_similarity(
                question_emb.reshape(1, -1),
                annotation.question_emb.reshape(1, -1)
            )[0][0]

            sim_s = cosine_similarity(
                step_emb.reshape(1, -1),
                annotation.step_emb.reshape(1, -1)
            )[0][0]

            # Weighted combination
            score = alpha * sim_q + (1 - alpha) * sim_s
            results.append((annotation, score))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:topk]
    
    def serialize(self, path: str = './'):
        """
        Serialize current memory into the specified path as 'memory.json'.
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)  # ensure directory exists

        memory_file = path / "memory.json"

        # Convert AnnotatedMemory objects to dicts
        memory_list = []
        for mem in self.memory:
            memory_list.append({
                "memory_id": getattr(mem, "memory_id", None),  # optional if you have it
                "question": mem.question,
                "step": mem.step,
                "experiences": mem.experiences,
                "question_emb": mem.question_emb,
                "step_emb": mem.step_emb
            })

        # Write to file
        with memory_file.open("w", encoding="utf-8") as f:
            json.dump(memory_list, f, ensure_ascii=False, indent=2)

        print(f"Memory serialized to {memory_file}")

    def load(self, path: str):
        """
        Load memory from a JSON file at the given path.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        # Step 1: load JSON from path
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Loaded JSON must be a list of AnnotatedMemory entries")

        # Step 2: process each loaded memory
        for mem_dict in data:
            # Convert dict to AnnotatedMemory instance
            mem_entry = AnnotatedMemory(
                question=mem_dict["question"],
                step=mem_dict["step"],
                experiences=mem_dict.get("experiences", []),
                question_emb=mem_dict["question_emb"],
                step_emb=mem_dict["step_emb"]
            )

            # Check if same question + step already exists
            existing = None
            for m in self.memory:
                if m.question == mem_entry.question and m.step == mem_entry.step:
                    existing = m
                    break

            if existing:
                # Merge experiences, remove duplicates
                combined = list(set(existing.experiences + mem_entry.experiences))
                existing.experiences = combined
            else:
                # Add new memory
                self.memory.append(mem_entry)

        print(f"Loaded {len(data)} memory entries from {path}")