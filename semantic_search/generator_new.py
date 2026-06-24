from ollama import chat
import time


class GeneratorAgent:
    def __init__(self):
        pass
    def _format_context(self, documents):
        context = ""

        for i, doc in enumerate(documents, start=1):
            context += f"[Document {i}]\n"
            context += doc
            context += "\n\n"
        return context

    def execute(self, query, documents):
        start_time = time.time()

        context = self._format_context(
            documents
        )

        prompt = f"""
Question:
{query}

Source Documents:
{context}
Rules:
- Answer ONLY using the provided documents
- Do NOT use outside knowledge
- If the documents do not contain enough information, say so
- Be clear and concise

Answer:
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

        generated_answer = (
            response["message"]["content"]
            .strip()
        )

        output = {
            "query": query,
            "generated_answer":
                generated_answer,
            "num_sources":
                len(documents),
            "execution_time":
                time.time() - start_time
        }
        return output