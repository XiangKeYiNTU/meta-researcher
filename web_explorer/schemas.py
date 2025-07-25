from pydantic import BaseModel, Field

class Step(BaseModel):
    goal: str = Field(..., description="The goal of the step.")
    instructions: str = Field(..., description="Instructions for how to achieve the goal.")

class Plan(BaseModel):
    steps: list[Step] = Field(..., description="List of steps in the plan.")
