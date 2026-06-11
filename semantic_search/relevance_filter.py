import time


class RelevanceFilter:

    def __init__(self, relevance_threshold=0.5):
        self.relevance_threshold = relevance_threshold

    def _calculate_relevance(
        self,
        query,
        document
    ):

        query_words = query.lower().split()
        document_lower = document.lower()

        matches = 0
        for word in query_words:
            if word in document_lower:
                matches += 1
            return matches / len(query_words)

    def execute(self, agent1_output):
        start_time = time.time()
        query = agent1_output["query"]
        filtered_documents = []

        for doc in agent1_output["retrieved_documents"]:
            relevance_score = self._calculate_relevance(
                query,
                doc["content"]
            )

            if relevance_score >= self.relevance_threshold:
                filtered_documents.append(
                    {
                        "content": doc["content"],
                        "source": doc["source"],
                        "confidence": doc["confidence"],
                        "relevance_score": relevance_score
                    }
                )

        output = {
            "query": query,
            "filtered_documents": filtered_documents,
            "num_filtered": len(filtered_documents),
            "execution_time": time.time() - start_time
        }

        return output