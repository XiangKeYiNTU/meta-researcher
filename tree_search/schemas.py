from pydantic import BaseModel, Field
from enum import Enum

class Step(BaseModel):
    goal: str = Field(..., description="The goal of the step.")
    instructions: str = Field(..., description="Instructions for how to achieve the goal.")

class ActionType(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"

class ActionParams(BaseModel):
    position: int = Field(..., description="The position in the plan where the action is applied.")
    goal: str = Field(..., description="The goal of the step to be added, removed, or updated.")
    instructions: str = Field(..., description="Instructions for how to achieve the goal.")

class ModificationResponse(BaseModel):
    rationale: str = Field(..., description="The rationale for the modification.")
    action: ActionType = Field(..., description="The type of action taken.")
    action_params: ActionParams = Field(..., description="Parameters for the action taken.")


class Plan(BaseModel):
    steps: list[Step] = Field(..., description="List of steps in the plan.")

    def apply_modifications(self, modifications: list[dict]) -> 'Plan':
        """
        Apply modifications to the plan based on the provided modifications.
        Each modification should contain 'rationale', 'action', and 'action_params'.
        """
        modified_plan = self.model_copy(deep=True)  # Create a deep copy of the plan

        if len(modifications) == 0:
            return modified_plan

        # This is a placeholder implementation. Actual logic will depend on how modifications are defined.
        for mod in modifications:
            if mod.action == ActionType.ADD:
                modified_plan.steps.insert(mod.action_params.position, Step(goal=mod.action_params.goal, instructions=mod.action_params.instructions))
            elif mod.action == ActionType.REMOVE:
                if 0 <= mod.action_params.position < len(modified_plan.steps):
                    modified_plan.steps.pop(mod.action_params.position)
            elif mod.action == ActionType.UPDATE:
                if 0 <= mod.action_params.position < len(modified_plan.steps):
                    modified_plan.steps[mod.action_params.position] = Step(goal=mod.action_params.goal, instructions=mod.action_params.instructions)
        # Return the modified plan
        return modified_plan

class PlanScore(BaseModel):
    effectiveness: int = Field(..., description="Effectiveness score of the plan.")
    completeness: int = Field(..., description="Completeness score of the plan.")
    executability: int = Field(..., description="Executability score of the plan.")