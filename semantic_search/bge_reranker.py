from sentence_transformers import CrossEncoder


class BGEReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

    def _batch_score(
        self,
        query,
        documents
    ):

        pairs = [
            (query, doc)
            for doc in documents
        ]

        scores = self.model.predict(
            pairs
        )

        return scores.tolist()

    def execute(
        self,
        query,
        documents,
        distances=None,
        ids=None
    ):

        scores = self._batch_score(
            query,
            documents
        )

        combined = []

        for i, doc in enumerate(
            documents
        ):

            combined.append({

                "document": doc,

                "score": scores[i],

                "distance":
                    distances[i]
                    if distances
                    else None,

                "id":
                    ids[i]
                    if ids
                    else None
            })

        combined.sort(
            key=lambda x:
                x["score"],
            reverse=True
        )

        return {

            "documents": [
                item["document"]
                for item in combined
            ],

            "scores": [
                item["score"]
                for item in combined
            ],

            "distances": [
                item["distance"]
                for item in combined
            ],

            "ids": [
                item["id"]
                for item in combined
            ],

            "num_documents":
                len(combined)
        }