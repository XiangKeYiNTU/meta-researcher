planning_prompt = """You are a task planner who gives step-by-step plans with detailed goals and instructions.

You are given:
- A question
- The question's ground truth answer

Your task:
- Provide a plan that is correct, efficient, and complete to reach to the ground truth answer.
- You can't directly give ground truth answer, or parts of ground truth answer, or hints of ground truth answer anywhere in the plan.
- Each step of the plan must contain a goal and its instruction.
- Each step must be executable within two searches and 10 times of browsing, no redundant or unnecessary steps.
- Output a JSON object as follows:
```JSON
{{
    "steps": [
        {{
            "goal": "Find TV shows from the 1960s-1980s with fewer than 50 episodes.",
            "instructions": "Search for short-run or cult classic TV shows from that era using terms like “TV shows 1960s-1980s under 50 episodes.” Focus on comedy or satire genres."
        }},
        {{
            "goal": "Identify characters from these shows who break the fourth wall and are humorous.",
            "instructions": "Search for “TV characters who break the fourth wall” and cross-check with the list of shows found. Prioritize comedic characters."
        }},
        {{
            "goal": "Check which of these characters has a backstory involving selfless ascetics.",
            "instructions": "Search for the origin or backstory of each character using terms like “Character name backstory monks” or “trained by ascetics.”"
        }},
        {{
            "goal": "Confirm the character matches all criteria.",
            "instructions": "Verify that the character is humorous, breaks the fourth wall, appeared in a show with <50 episodes in the 1960s-1980s, and has an ascetic-related backstory."
        }}
    ]
}}
```
---
Question: {}
Ground truth answer: {}
"""