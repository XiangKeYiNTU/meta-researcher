from pydantic import BaseModel, Field
from typing import List

class AnnotatedMemory(BaseModel):
    question: str = Field(..., description="The question to be solved of this step.")
    step: str = Field(..., description="The goal of the step.")
    experiences: List[str] = Field(..., description="Annotated experiences from the history of step execution.")
    question_emb: List[float] = Field(..., description="Embedding for question.")
    step_emb: List[float] = Field(..., description="Embedding for step.")
