import time

from embeddings_real import (
    load_documents,
    create_embeddings,
    create_chroma_collection,
)


class RetrievalAgent:
    def __init__(self, collection, model, top_k=5):
        self.collection = collection
        self.model = model
        self.top_k = top_k

    def _convert_distance_to_confidence(self, distance):
        return max(0.0, min(1.0, 1.0 - distance))

    def validate_output(self, output):
        required_keys = [
            "query",
            "retrieved_documents",
            "num_results",
            "execution_time",
        ]

        for key in required_keys:
            if key not in output:
                raise ValueError(f"Missing key: {key}")

        if not isinstance(output["retrieved_documents"], list):
            raise ValueError("retrieved_documents must be a list")

        if output["execution_time"] <= 0:
            raise ValueError("execution_time must be positive")

        for doc in output["retrieved_documents"]:
            required_doc_keys = ["content", "source", "distance", "confidence"]
            for key in required_doc_keys:
                if key not in doc:
                    raise ValueError(f"Missing document key: {key}")

            if doc["distance"] < 0:
                raise ValueError("Invalid distance")

            if not (0 <= doc["confidence"] <= 1):
                raise ValueError("Invalid confidence")

        return True

    def execute(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.time()

        query_embedding = self.model.encode(query, convert_to_numpy=True)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=self.top_k
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if len(documents) == 0:
            output = {
                "query": query,
                "retrieved_documents": [],
                "num_results": 0,
                "execution_time": time.time() - start_time,
            }

            return output

        retrieved_documents = []

        for content, doc_id, distance in zip(documents, ids, distances):
            confidence = self._convert_distance_to_confidence(distance)

            retrieved_documents.append(
                {
                    "content": content,
                    "source": doc_id,
                    "distance": float(distance),
                    "confidence": float(confidence),
                }
            )

        output = {
            "query": query,
            "retrieved_documents": retrieved_documents,
            "num_results": len(retrieved_documents),
            "execution_time": time.time() - start_time,
        }

        self.validate_output(output)

        return output


def main():
    print("\n RETRIEVAL AGENT\n")

    docs, doc_ids = load_documents(
        "data/FastAPI_and_Scikit-learn_Technical_Guide.pdf"
    )

    model, embeddings = create_embeddings(docs)

    collection = create_chroma_collection(docs, embeddings, doc_ids)

    agent = RetrievalAgent(collection, model, top_k=5)

    query = input("\nAsk a question: ")

    result = agent.execute(query)

    print(f"\nFound {result['num_results']} documents\n")

    for i, doc in enumerate(result["retrieved_documents"], start=1):
        print(f"{i}. [Confidence: {doc['confidence']:.2f}] {doc['source']}")
        print(doc["content"][:250])
        print("-" * 60)


if __name__ == "__main__":
    main()
