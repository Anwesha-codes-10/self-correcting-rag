import time


class Generator:

    def execute(self, agent2_output):

        start_time = time.time()

        query = agent2_output["query"]

        filtered_documents = (
            agent2_output["filtered_documents"]
        )

        answer = []

        for doc in filtered_documents:

            answer.append(
                doc["content"]
            )

        generated_answer = "\n\n".join(
            answer
        )

        output = {
            "query": query,
            "generated_answer": generated_answer,
            "sources": [
                doc["source"]
                for doc in filtered_documents
            ],
            "execution_time":
                time.time() - start_time
        }

        return output