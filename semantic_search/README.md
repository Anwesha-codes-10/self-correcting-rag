# 🔎 Semantic Search Fundamentals

Part of the **Self-Correcting RAG** project.

## 🎯 Why This Exists

Before using frameworks like LangChain, ChromaDB, FAISS, or Pinecone, I wanted to understand how semantic retrieval works under the hood.

This module implements the mathematical foundation of semantic search from scratch using NumPy.

Instead of relying on libraries, the focus here is understanding:

* How vectors represent meaning
* How cosine similarity measures semantic closeness
* How retrieval systems rank documents
* How Top-K retrieval works in modern RAG pipelines

---

## 🧠 What I Built

### Cosine Similarity from Scratch

Implemented:

\cos(\theta)=\frac{A\cdot B}{|A||B|}

This measures how similar two vectors are by comparing their direction.

---

### Document Retrieval Engine

Given:

* Query embedding
* Document embeddings

The system:

1. Computes similarity scores
2. Ranks all documents
3. Returns the Top-K most relevant results

---

### Unit Tests

Implemented tests for:

✅ Identical vectors

✅ Orthogonal vectors

✅ Partial similarity

✅ Zero-vector edge cases

✅ Shape mismatch validation

✅ Retrieval ranking

✅ Top-K document retrieval

---

## 📂 Project Structure

```text
semantic_search/
│
├── embeddings.py
├── test_embeddings.py
└── README.md
```

---

## ⚙️ Retrieval Workflow

```text
User Query
     │
     ▼
Query Embedding
     │
     ▼
Cosine Similarity
     │
     ▼
Compare Against All Documents
     │
     ▼
Similarity Scores
     │
     ▼
Ranking
     │
     ▼
Top-K Results
```

---

## 🚀 Example

Query:

```text
How do I do dependency injection in FastAPI?
```

Retrieved:

```text
1. FastAPI uses Depends() for dependency injection
2. You can inject dependencies into FastAPI route handlers
3. Django has a different dependency system
```

---

## 📚 Key Concepts Learned

* Vector Representation
* Dot Product
* Vector Magnitude (L2 Norm)
* Cosine Similarity
* Ranking Algorithms
* Top-K Retrieval
* Defensive Programming
* Unit Testing

---

## 🛣️ Roadmap

### Completed

* [x] Byte Pair Encoding (BPE)
* [x] Cosine Similarity
* [x] Semantic Search Fundamentals
* [x] Retrieval Testing

### Next

* [ ] SentenceTransformer Embeddings
* [ ] Document Chunking
* [ ] Chroma Vector Database
* [ ] Retrieval-Augmented Generation (RAG)
* [ ] Hallucination Detection
* [ ] Self-Correction Module

---

## 💡 Takeaway

Most tutorials start with frameworks.

This project starts with fundamentals.

The goal is not only to use RAG systems, but to understand and debug them from first principles.

