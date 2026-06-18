from ollama import chat


class QueryExpansion:

    def execute(self, query):

        prompt = f"""
Generate exactly 2 alternative ways to ask the same question.

Question:
{query}

Rules:
- Keep the same meaning
- Return only the 2 rewritten questions
- One question per line
"""

        response = chat(
            model="qwen3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        output_text = response["message"]["content"]

        variations = []

        for line in output_text.split("\n"):
            line = line.strip()

            if line:
                variations.append(line)

        queries = [query]

        queries.extend(variations[:2])

        return queries