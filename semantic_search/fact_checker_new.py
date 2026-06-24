from ollama import chat
import time


class FactChecker:
    def execute(self, generated_answer, source_documents):
        start_time = time.time()
        context = "\n\n".join(source_documents)

        prompt = f"""
Answer:
{generated_answer}

Source Documents:
{context}

Determine whether the answer is fully supported by the source documents.

Return ONLY one word:

CONSISTENT

or

HALLUCINATION
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

        verdict = (
            response["message"]["content"]
            .strip()
            .upper()
        )

        confidence = (
            1.0
            if "CONSISTENT" in verdict
            else 0.0
        )

        return {
            "verdict":
                verdict,
            "confidence":
                confidence,
            "execution_time":
                time.time() - start_time
        }