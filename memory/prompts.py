annotation_prompt = """You are an expert that summarizes execution history into pieces of learnable experiences.

You are given:
- A question
- An intermediate step to solve the question
- An execution history of the step

Your task:
- Summarize the experience into learnable experiences

Instructions:
- A piece of experience is a short summary telling the executor what to be cautious of in every execution process.
- The experiences must be general and applicable to all problems involving information seeking.
- Every experience must be prefixed with an index

Example output:
1. Focus on sources that ...
2. Don't ...
3. ...
"""