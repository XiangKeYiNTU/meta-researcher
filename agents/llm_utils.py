import re
from typing import Tuple, Optional

def extract_chosen_index(response: str) -> Tuple[str, int]:
    """
    Extracts the chosen step index from the response string.

    Parameters:
        response (str): The response string containing the chosen step.

    Returns:
        Tuple[str, int]: Whether the extraction is successful and the index of the chosen step.
    """
    index = re.search(r"<choose>(.*?)</choose>", response, re.DOTALL)

    if not index:
        # raise ValueError("Response does not contain valid <choose> markers.")
        return ("Response does not contain valid <choose> markers.", -1)

    chosen_step_str = index.group(1).strip()

    try:
        return ("success", int(chosen_step_str))
    except ValueError:
        # raise ValueError(f"Invalid step number: {chosen_step_str}")
        return ("The step number provided is invalid, please provide your chosen step index between `<choose>` and `<choose>`", -1)
    
def extract_finalized_answer(response: str) -> Tuple[str, Optional[str]]:
    """
    Extracts the finalized answer from the response string.

    Parameters:
        response (str): The response string containing the finalized answer.

    Returns:
        Tuple[str, Optional[str]]: A tuple containing (status, content) where:
            - status: "success" if extraction succeeded, error message otherwise
            - content: The extracted content if successful, None otherwise
    """
    index = re.search(r"<finalize>(.*?)</finalize>", response, re.DOTALL)

    if not index:
        # raise ValueError("Response does not contain valid <finalize> markers.")
        return ("Please provide your response between `<finalize>` and `</finalize>`", None)

    return ("success", index.group(1).strip())