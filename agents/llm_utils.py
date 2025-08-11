import re

def extract_chosen_index(response: str) -> int:
    """
    Extracts the chosen step index from the response string.

    Parameters:
        response (str): The response string containing the chosen step.

    Returns:
        int: The index of the chosen step.
    """
    index = re.search(r"<choose>(.*?)</choose>", response, re.DOTALL)

    if not index:
        raise ValueError("Response does not contain valid <choose> markers.")

    chosen_step_str = index.group(1).strip()

    try:
        return int(chosen_step_str)
    except ValueError:
        raise ValueError(f"Invalid step number: {chosen_step_str}")
    
def extract_finalized_answer(response: str) -> str:
    """
    Extracts the finalized answer from the response string.

    Parameters:
        response (str): The response string containing the finalized answer.

    Returns:
        str: The finalized answer.
    """
    index = re.search(r"<finalize>(.*?)</finalize>", response, re.DOTALL)

    if not index:
        raise ValueError("Response does not contain valid <finalize> markers.")

    return index.group(1).strip()