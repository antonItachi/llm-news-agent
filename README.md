# RAG Agent Framework

This project is a minimal retrieval-augmented generation (RAG) framework with a custom agent architecture and integrated tool system.

## Overview

- **Agent-based system** handling multi-step queries with tool usage and memory tracking.
- **Custom tools** including:
  - `news_api_search`: fetches and parses recent news articles.
  - `rag_search`: performs similarity search over news content using embedding-based vector store (Qdrant).
- **Session memory tracker** with in-memory buffer per session for conversation continuity.
- **Multi-query reformulation** step allows the model to generate diverse retrieval prompts for more relevant results.
- **Manual embedding flow** using chosen transformer model; no auto-wrapping or hidden processes.
- **Embedding model and chunking** configured explicitly.
- **No third-party frameworks** like LangChain or Haystack — all logic is handled internally.

This project is built for full control and inspection over each step of the retrieval-generation loop. It acts as a personal assistant framework with tool orchestration, memory, and query planning.

---

The system can be extended with additional tools, storage backends, or optimized LLM backends.
