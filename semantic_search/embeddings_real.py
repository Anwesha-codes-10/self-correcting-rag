from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb


def load_documents(pdf_path):
    reader = PdfReader(pdf_path)

    docs = []
    doc_ids = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            docs.append(text.strip())
            doc_ids.append(
                f"PAGE_{page_number + 1}"
            )

    return docs, doc_ids


def create_embeddings(
    docs,
    model_name="all-MiniLM-L6-v2"
):
    model = SentenceTransformer(
        model_name
    )

    embeddings = model.encode(
        docs,
        convert_to_numpy=True
    )

    return model, embeddings


def create_chroma_collection(
    docs,
    embeddings,
    doc_ids,
    collection_name="fastapi_docs"
):
    client = chromadb.Client()

    try:
        client.delete_collection(
            collection_name
        )
    except:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        ids=doc_ids
    )

    return collection


def query_documents(
    collection,
    model,
    query_text,
    top_k=3
):
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return list(
        zip(
            documents,
            distances
        )
    )


def main():

    print("\nLoading PDF...")

    docs, doc_ids = load_documents(
        "semantic_search/data/FastAPI_and_Scikit-learn_Technical_Guide.pdf"
    )

    print(
        f"Loaded {len(docs)} pages"
    )

    print(
        "\nCreating embeddings..."
    )

    model, embeddings = (
        create_embeddings(docs)
    )

    print(
        f"Embeddings Shape: {embeddings.shape}"
    )

    print(
        "\nCreating Chroma Collection..."
    )

    collection = (
        create_chroma_collection(
            docs,
            embeddings,
            doc_ids
        )
    )

    print(
        f"Stored {collection.count()} documents"
    )

    query = input(
        "\nAsk a question: "
    )

    results = query_documents(
        collection,
        model,
        query,
        top_k=3
    )

    print("\nTop Results:\n")

    for i, (
        doc,
        distance
    ) in enumerate(
        results,
        start=1
    ):
        print(
            f"{i}. Distance: {distance:.4f}"
        )

        print(
            doc[:300]
        )

        print(
            "-" * 60
        )


if __name__ == "__main__":
    main()