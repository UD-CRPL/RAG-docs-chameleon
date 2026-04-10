"""
Run this script once to build the FAISS vector index from Chameleon docs.
Re-run whenever the docs change or you want to refresh the index.

Usage:
    python build_index.py
"""
from loader import loader_docs
from rag import split_docs, create_vectorstore, VECT_STORE_PATH

if __name__ == "__main__":
    print("Loading docs...")
    docs = loader_docs()
    print(f"Loaded {len(docs)} documents.")

    chunks = split_docs(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("Building vector store...")
    create_vectorstore(chunks)
    print(f"Index saved to '{VECT_STORE_PATH}'. Done.")
