import numpy as np

from embeddings_real import (
    load_documents,
    create_embeddings,
    create_chroma_collection,
    query_documents
)


PDF_PATH = (
    "semantic_search/data/FastAPI_and_Scikit-learn_Technical_Guide.pdf"
)


def test_load_documents():

    docs, doc_ids = (
        load_documents(
            PDF_PATH
        )
    )

    assert len(docs) > 0

    assert len(docs) == len(doc_ids)

    print(
        "✓ test_load_documents PASSED"
    )


def test_create_embeddings():

    docs = [
        "FastAPI routes",
        "Dependency injection",
        "Random Forest"
    ]

    model, embeddings = (
        create_embeddings(docs)
    )

    assert (
        embeddings.shape[0] == 3
    )

    assert (
        embeddings.shape[1] == 384
    )

    print(
        "✓ test_create_embeddings PASSED"
    )


def test_similarity():

    docs = [
        "FastAPI routes",
        "Create API routes",
        "Random Forest Classifier"
    ]

    model, embeddings = (
        create_embeddings(docs)
    )

    sim1 = np.dot(
        embeddings[0],
        embeddings[1]
    )

    sim2 = np.dot(
        embeddings[0],
        embeddings[2]
    )

    assert sim1 > sim2

    print(
        "✓ test_similarity PASSED"
    )


def test_chroma_collection():

    docs = [
        "FastAPI",
        "Pydantic",
        "Scikit Learn"
    ]

    ids = [
        "DOC1",
        "DOC2",
        "DOC3"
    ]

    model, embeddings = (
        create_embeddings(docs)
    )

    collection = (
        create_chroma_collection(
            docs,
            embeddings,
            ids,
            "test_collection"
        )
    )

    assert (
        collection.count() == 3
    )

    print(
        "✓ test_chroma_collection PASSED"
    )


def test_query_documents():

    docs = [
        "FastAPI routes",
        "Dependency injection",
        "Random Forest"
    ]

    ids = [
        "DOC1",
        "DOC2",
        "DOC3"
    ]

    model, embeddings = (
        create_embeddings(docs)
    )

    collection = (
        create_chroma_collection(
            docs,
            embeddings,
            ids,
            "query_collection"
        )
    )

    results = query_documents(
        collection,
        model,
        "How to create routes?",
        top_k=2
    )

    assert len(results) > 0

    print(
        "✓ test_query_documents PASSED"
    )


if __name__ == "__main__":

    test_load_documents()
    test_create_embeddings()
    test_similarity()
    test_chroma_collection()
    test_query_documents()

    print(
        "\n✅ ALL TESTS PASSED"
    )