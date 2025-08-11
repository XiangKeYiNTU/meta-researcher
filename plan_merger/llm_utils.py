from plan_merger.base import StepNode
from typing import List

def extract_next_steps(node_list: List[StepNode]):
    steps = ""
    for i, node in enumerate(node_list):
        steps += f"{i}. {node.step.goal}\nInstruction: {node.step.instructions}\n\n"

    return steps
        
        
