RAG-docs-chameleon
--------------------------------------------------------------------------------------------------
Overview
This repository provides a lightweight Retrieval-Augmented Generation (RAG) pipeline for answering questions about Chameleon Cloud documentation, plus simple structure-similarity metrics for quick evaluation (Jaccard and Cosine). The repo has been trimmed to the essentials—experimental files are removed—so you can build an index, ask questions with citations, and score results with minimal setup.
What you can do
Ingest & sanitize docs → build an embedding index (loader.py)
Ask questions via CLI → retrieve top-k chunks → generate answers with citations (rag.py)
Compute quick structure-similarity metrics (Jaccard/Cosine) for your predictions (integrated hooks)
