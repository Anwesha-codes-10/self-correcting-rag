from retrieval_agent import RetrievalAgent
from relevance_filter import RelevanceFilter
from generator import Generator

from embeddings_real import (
    load_documents,
    create_embeddings,
    create_chroma_collection
)

def main():
    docs, doc_ids = load_documents(
        "data/FastAPI_and_Scikit-learn_Technical_Guide.pdf"
    )

    model, embeddings = create_embeddings(docs)
    collection = create_chroma_collection(
        docs,
        embeddings,
        doc_ids
    )

    agent1 = RetrievalAgent(
        collection,
        model,
        top_k=5
    )

    agent2 = RelevanceFilter(
        relevance_threshold=0.4
    )

    agent3 = Generator()

    query = input(
        "\nAsk a question: "
    )

    print("\n--- AGENT 1 ---")
    result1 = agent1.execute(query)
    print(
        f"Retrieved {result1['num_results']} documents"
    )

    print("\n--- AGENT 2 ---")
    result2 = agent2.execute(result1)
    print(
        f"Filtered to {result2['num_filtered']} relevant documents\n"
    )

    print("\n--- AGENT 3 ---")
    result3 = agent3.execute(result2)
    print(
        f"Generated answer:\n{result3['generated_answer']}\n"
    )

    print("Sources:")
    for source in result3["sources"]:
        print(source)

    for i, doc in enumerate(
        result2["filtered_documents"],
        start=1
    ):

        print(
            f"{i}. Relevance: {doc['relevance_score']:.2f}"
        )

        print(
            doc["content"][:250]
        )

        print("-" * 60)


if __name__ == "__main__":
    main()