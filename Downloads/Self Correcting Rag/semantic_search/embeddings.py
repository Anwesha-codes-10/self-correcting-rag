import numpy as np

def cosine_similarity(vec1, vec2):
    if vec1.shape != vec2.shape:
        raise ValueError(
            f"Shape mismatch: {vec1.shape} vs {vec2.shape}"
        )

    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0

    return dot_product / (magnitude_vec1 * magnitude_vec2)


def retrieve_documents(
    query_embedding,
    doc_embeddings,
    doc_texts,
    top_k=3
):
    similarities = np.array([
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in doc_embeddings
    ])

    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        (doc_texts[idx], similarities[idx])
        for idx in sorted_indices
    ]
