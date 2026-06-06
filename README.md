# Self-Correcting RAG

Self-Correcting RAG is an exploration of Retrieval-Augmented Generation (RAG) systems and Large Language Models (LLMs), built from the ground up to understand how modern AI applications retrieve information, generate responses, and improve answer quality through self-correction.

The project aims to combine retrieval techniques, language models, and verification mechanisms into a unified pipeline capable of producing more reliable and context-aware responses.

## Objectives

* Understand the internal workings of Large Language Models.
* Implement core NLP and LLM concepts from scratch.
* Explore tokenization, embeddings, attention mechanisms, and transformers.
* Build a Retrieval-Augmented Generation pipeline.
* Develop methods to detect and reduce hallucinations.
* Implement self-correction strategies to improve generated responses.

## Planned Architecture

```text
User Query
     ↓
Document Retrieval
     ↓
Context Selection
     ↓
Response Generation
     ↓
Answer Verification
     ↓
Self-Correction
     ↓
Final Response
```

## Technologies

* Python
* PyTorch
* Tiktoken
* Hugging Face
* FAISS
* LangChain
* Streamlit

## Vision

The long-term goal is to build an end-to-end self-correcting AI assistant that can retrieve relevant information, evaluate its own outputs, and iteratively improve response quality before presenting an answer to the user.
