class MultiQueryRetrieval:

    def __init__(self, collection, model):
        self.collection = collection
        self.model = model

    def execute(self, query_list):

        results = {}

        for query in query_list:
            query_embedding = self.model.encode(query).tolist()

            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=7
            )

            documents = search_results["documents"][0]
            distances = search_results["distances"][0]

            ids = search_results["ids"][0]

            for doc_id, doc, distance in zip(ids, documents, distances):
                if doc_id not in results:
                    results[doc_id] = {
                        "document": doc,
                        "distance": distance
                    }

                else:
                    results[doc_id]["distance"] = min(
                        results[doc_id]["distance"],
                        distance
                    )

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["distance"]
        )

        sorted_results = sorted_results[:10]

        merged_documents = []
        merged_distances = []
        merged_ids = []

        for doc_id, data in sorted_results:

            merged_ids.append(doc_id)
            merged_documents.append(data["document"])
            merged_distances.append(data["distance"])

        return {
            "documents": merged_documents,
            "distances": merged_distances,
            "ids": merged_ids,
            "num_documents": len(merged_documents)
        }