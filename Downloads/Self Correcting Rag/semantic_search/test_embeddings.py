import numpy as np
from embeddings import cosine_similarity, retrieve_documents

def test_identical_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])

    assert cosine_similarity(v1, v2) == 1.0

def test_orthogonal_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    assert cosine_similarity(v1, v2) == 0.0

def test_partial_similarity():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 1.0])

    similarity = cosine_similarity(v1, v2)

    assert 0.70 < similarity < 0.72

def test_zero_vector():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])

    assert cosine_similarity(v1, v2) == 0.0

def test_shape_mismatch():
    try:
        cosine_similarity(
            np.array([1, 2]),
            np.array([1, 2, 3])
        )

        assert False

    except ValueError:
        pass

def test_document_ranking():
    docs = ["doc1", "doc2", "doc3"]

    embeddings = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.5, 0.5]
    ])

    query = np.array([0.95, 0.05])

    results = retrieve_documents(
        query,
        embeddings,
        docs
    )

    assert results[0][0] == "doc1"

def test_top_k():
    docs = ["doc1", "doc2", "doc3", "doc4"]

    embeddings = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8]
    ])

    query = np.array([0.95, 0.05])

    results = retrieve_documents(
        query,
        embeddings,
        docs,
        top_k=2
    )

    assert len(results) == 2
