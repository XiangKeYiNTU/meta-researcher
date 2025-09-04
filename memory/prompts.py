annotation_prompt = """You are an expert that summarizes execution history into pieces of learnable experiences.

You are given:
- A question
- An intermediate step to solve the question
- An execution history of the step

Your task:
- Summarize the experience into learnable experiences

Instructions:
- The execution process has flaws and loopholes most of the time and is not always a success.
- A piece of experience is a short summary telling the executor what to be cautious of in every execution process.
- Come up with instructions within the experiences so the executor won't make the same or similar mistakes.
- The experiences must be general and applicable to all problems involving information seeking.
- Every experience must be prefixed with an index.
- No more than 3 general experiences.

Example output:
1. Focus on sources that ...
2. Don't ...
3. ...
"""