import time


class FactChecker:

    def execute(
        self,
        agent3_output,
        agent2_output
    ):

        start_time = time.time()

        generated_answer = (
            agent3_output["generated_answer"]
            .lower()
        )

        source_text = ""

        for doc in agent2_output[
            "filtered_documents"
        ]:

            source_text += (
                doc["content"].lower()
                + " "
            )

        answer_words = (
            generated_answer.split()
        )

        matches = 0

        for word in answer_words:

            if word in source_text:
                matches += 1

        confidence = (
            matches /
            max(len(answer_words), 1)
        )

        is_grounded = (
            confidence >= 0.7
        )

        output = {
            "query":
                agent3_output["query"],

            "generated_answer":
                agent3_output[
                    "generated_answer"
                ],

            "is_grounded":
                is_grounded,

            "confidence":
                confidence,

            "execution_time":
                time.time() - start_time
        }

        return output