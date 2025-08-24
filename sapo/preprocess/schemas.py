from pydantic import BaseModel, Field

class Step(BaseModel):
    goal: str = Field(..., description="The goal of the step.")
    instructions: str = Field(..., description="Instructions for how to achieve the goal.")
    # max_reward: float = Field(default=0.0, description="Current maximum execution reward.")
    # execution_result: str = Field(default="", description="Current execution result")

class Plan(BaseModel):
    steps: list[Step] = Field(..., description="List of steps in the plan.")

class Data(BaseModel):
    question: str = Field(..., description="The question to solve.")
    gt_answer: str = Field(..., description="Ground truth answer for the question.")
    plans: list[Plan] = Field(..., description="Generated plans for the question")

class Dataset(BaseModel):
    data: list[Data] = Field(..., description="Current collected training data.")
    cur_index: int = Field(default=0, description="Current collected index.")