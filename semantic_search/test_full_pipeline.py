from query_expansion import QueryExpansion
from multi_query_retrieval import MultiQueryRetrieval
from bge_reranker import BGEReranker
from llm_relevance_filter import LLMRelevanceFilter
from generator_new import GeneratorAgent
from fact_checker_new import FactChecker

from embeddings_real import (
    load_documents,
    create_embeddings,
    create_chroma_collection
)


print("\nLoading documents...")

docs, doc_ids = load_documents(
    "data/FastAPI_and_Scikit-learn_Technical_Guide.pdf"
)

model, embeddings = create_embeddings(
    docs
)

collection = create_chroma_collection(
    docs,
    embeddings,
    doc_ids
)

print("System Ready")


expander = QueryExpansion()

retriever = MultiQueryRetrieval(
    collection,
    model
)

reranker = BGEReranker()

filter_agent = LLMRelevanceFilter()

generator = GeneratorAgent()

checker = FactChecker()


query = input(
    "\nAsk Question: "
)


# Stage 1
query_list = expander.execute(
    query
)

print("\nExpanded Queries:")
for q in query_list:
    print("-", q)


# Stage 2
retrieval_results = retriever.execute(
    query_list
)

print(
    f"\nRetrieved "
    f"{retrieval_results['num_documents']} docs"
)


# Stage 3
reranked_results = reranker.execute(
    query,
    retrieval_results["documents"],
    retrieval_results["distances"],
    retrieval_results["ids"]
)

print(
    f"\nReranked "
    f"{reranked_results['num_documents']} docs"
)


# Stage 4
filtered_results = filter_agent.execute(
    query,
    reranked_results["documents"][:5],
    reranked_results["scores"][:5],
    reranked_results["ids"][:5]
)

print(
    f"\nFiltered to "
    f"{filtered_results['num_documents']} docs"
)


# Stage 5
generator_results = generator.execute(
    query,
    filtered_results["documents"]
)

print("\nGenerated Answer:\n")

print(
    generator_results[
        "generated_answer"
    ]
)


# Stage 6
fact_results = checker.execute(
    generator_results[
        "generated_answer"
    ],
    filtered_results[
        "documents"
    ]
)

print("\nFact Check:\n")

print(
    fact_results
)