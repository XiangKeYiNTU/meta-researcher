from transformers import pipeline, TextStreamer

from tree_search.schemas import Plan, Step
from plan_merger.base import PlanGraph
# from plan_merger.llm_utils import extract_next_steps
from agents.prompts import choose_prompt, finalize_prompt
from agents.llm_utils import extract_chosen_index, extract_finalized_answer

class MetaAgent:
    def __init__(self, plan_graph: PlanGraph, question: str, generator: pipeline, streamer: TextStreamer):
        self.plan_graph = plan_graph
        self.question = question
        self.generator = generator
        self.streamer = streamer

    def generate_next_step(self) -> Step:
        """
        let meta agent choose the next step to take
        """
        # Build up user prompt

        previous_step_results = self.plan_graph.get_current_exec_results()

        if len(previous_step_results) == 0:
            previous_steps = "No previously finished steps yet, you are choosing the first step to take."
        else:
            previous_steps = ""
            for step, result in previous_step_results:
                previous_steps += f"Step: {step.goal}\nAnswer: {result}\n\n"

        next_candidates = self.plan_graph.get_next_exec_steps()
        if len(next_candidates) == 1:
            return next_candidates[0].step

        next_steps = ""
        for i, step_node in enumerate(next_candidates):
            if step_node.step.goal == "END":
                return step_node.step
            next_steps += f"Candidate Step {i + 1}:\nGoal: {step_node.step.goal}\nInstructions: {step_node.step.instructions}\n\n"

        user_prompt = f"Question: {self.question}\n\nPrevious Steps:\n{previous_steps}\n\nNext Steps:\n{next_steps}"

        # response = self.openai_client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": choose_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )

        messages = [
            {"role": "system", "content": choose_prompt},
            {"role": "user", "content": user_prompt}
        ]
        while True:
            messages = self.generator(messages, max_new_tokens=1024, streamer=self.streamer)[0]["generated_text"]

            message, chosen_step = extract_chosen_index(messages[-1]["content"])
            if not chosen_step:
                # Meta agent didn't choose a step, update all current candidates and skip them
                for candidate_node in next_candidates:
                    real_node = self.plan_graph.exist_step(candidate_node.step)
                    real_node.execution_result = "Skipped by meta agent."
                return None
            if chosen_step == -1:
                messages.append({"role": "user", "content": message})
                continue
            else:
                return next_candidates[chosen_step - 1].step
    
    def finalize_answer(self) -> str:
        """
        Finalizes the answer to the question by considering all previous steps and their results.
        """
        previous_step_results = self.plan_graph.get_current_exec_results()
        previous_steps = ""
        for step, result in previous_step_results:
            previous_steps += f"Step: {step.goal}\nAnswer: {result}\n\n"

        user_prompt = f"Question: {self.question}\n\nPrevious Step Results:\n{previous_steps}"

        # response = self.openai_client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role":"system", "content": finalize_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )

        messages = [
            {"role":"system", "content": finalize_prompt},
            {"role": "user", "content": user_prompt}
        ]

        while True:
            messages = self.generator(messages, max_new_tokens=1024, streamer=self.streamer)[0]["generated_text"]

            message, answer = extract_finalized_answer(messages[-1]["content"])
            if answer:
                return answer
            else:
                messages.append({"role": "user", "content": message})
                continue