from ollama import chat
import re

class LLMRelevanceFilter:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def _judge_relevance(self, query, document):
        prompt = f"""
Question:
{query}

Document:
{document}

Is this document relevant to answering the question?

Rate relevance from 0.0 to 1.0

0.0 = completely irrelevant
1.0 = highly relevant

Return ONLY a number.
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

        output = response["message"]["content"].strip()

        match = re.search(r"\d*\.?\d+",output)

        if match:
            return float(match.group())
        return 0.0

    def execute(self, query, documents, scores=None, ids=None):
        filtered_docs = []
        relevance_scores = []
        filtered_ids = []
        filtered_bge_scores = []

        for i, doc in enumerate(documents):
            relevance = (self._judge_relevance(query, doc))

            if relevance >= self.threshold:
                filtered_docs.append(doc)

                relevance_scores.append(relevance)

                if ids:
                    filtered_ids.append(ids[i])

                if scores:
                    filtered_bge_scores.append(scores[i])

        return {
            "documents": filtered_docs,
            "relevance_scores": relevance_scores,
            "ids": filtered_ids,
            "bge_scores": filtered_bge_scores,
            "num_documents": len(filtered_docs)
        }
            