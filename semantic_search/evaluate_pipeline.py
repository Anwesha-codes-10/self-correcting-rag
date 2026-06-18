from retrieval_agent import RetrievalAgent
from relevance_filter import RelevanceFilter
from generator import Generator
from fact_checker import FactChecker

from embeddings_real import (
    load_documents,
    create_embeddings,
    create_chroma_collection
)


# Load documents
docs, doc_ids = load_documents(
    "data/FastAPI_and_Scikit-learn_Technical_Guide.pdf"
)

# Create embeddings
model, embeddings = create_embeddings(docs)

# Create ChromaDB collection
collection = create_chroma_collection(
    docs,
    embeddings,
    doc_ids
)

# Initialize agents
agent1 = RetrievalAgent(
    collection,
    model,
    top_k=5
)

agent2 = RelevanceFilter(
    relevance_threshold=0.4
)

agent3 = Generator()

agent4 = FactChecker()


test_queries = [
    "What is FastAPI?",
    "What are FastAPI routes?",
    "What is dependency injection?",
    "How does FastAPI handle errors?",
    "What is Pydantic?",
    "How do I create API endpoints?",
    "What are FastAPI key features?",
    "How can FastAPI be used with machine learning?",
    "How does request validation work?",
    "What deployment practices are recommended?"
]


successful = 0

for query in test_queries:

    print("\n" + "=" * 60)

    print(f"QUESTION: {query}")

    result1 = agent1.execute(query)

    result2 = agent2.execute(result1)

    result3 = agent3.execute(result2)

    result4 = agent4.execute(
        result3,
        result2
    )

    print(
        f"Retrieved: {result1['num_results']}"
    )

    print(
        f"Filtered: {result2['num_filtered']}"
    )

    print(
        f"Grounded: {result4['is_grounded']}"
    )

    print(
        f"Confidence: {result4['confidence']:.2f}"
    )

    if result2["num_filtered"] > 0:
        successful += 1

print("\n" + "=" * 60)

print(
    f"Successful Queries: "
    f"{successful}/{len(test_queries)}"
)

accuracy = (
    successful /
    len(test_queries)
) * 100

print(
    f"Pipeline Success Rate: "
    f"{accuracy:.2f}%"
)

print("=" * 60)