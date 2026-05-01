# Chameleon Docs Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the [Chameleon Cloud](https://chameleoncloud.org) testbed. It indexes documentation from multiple official sources and serves a Streamlit web UI backed by a Llama 3.3 70B language model via the [Tejas AI](https://ai.tejas.tacc.utexas.edu) API.

## Architecture

**Indexing** — `build_index.py` crawls documentation from ReadTheDocs, GitBook, the Chameleon blog, the community forum, and chameleoncloud.org. Each source is fetched independently and checkpointed to `_fetch_cache/` before anything is embedded, so a single source failure doesn't abort the run. Documents are split into large parent chunks (~2000 chars) for context and small child chunks (~400 chars) for embedding. Each child chunk is prefixed with a `[source_type: page title]` header so the embedding captures document-level context. Child chunks are indexed in a FAISS vector store; parent content is stored in `parents.json`. The new index is built in a temporary directory and atomically swapped into place only after the full build succeeds.

**Retrieval** — Incoming questions are embedded with E5-Mistral-7B-Instruct and matched against child chunks using MMR search. Results are deduplicated by parent ID and passed through a source-priority filter: readthedocs and blog chunks fill context slots first, with specialized sources (forum, gitbook, etc.) only included if slots remain. The corresponding parent chunks are then assembled as context for the LLM.

**Generation** — Meta-Llama-3.3-70B-Instruct generates answers strictly from the retrieved context. Conversation history (last 3 turns) is included for multi-turn use.

**Deployment** — Docker Compose runs the Streamlit app behind a Traefik reverse proxy with automatic TLS.

## Prerequisites

- Docker and Docker Compose
- A [Tejas AI](https://ai.tejas.tacc.utexas.edu) API key
- A domain name (for TLS via Let's Encrypt)

## Setup

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd RAG-docs-chameleon
    ```

2. Create a `.env` file in the root directory:

    ```
    TEJAS_API_KEY=<your_tejas_api_key>
    HOST=<your_domain>
    EMAIL=<your_email_for_letsencrypt>
    ```

3. Start the stack:

    ```bash
    docker compose up -d
    ```

4. Build the vector index (first run only, or after doc updates):

    ```bash
    docker exec rag-docs-chameleon-rag-app-1 python build_index.py
    ```

    This fetches all documentation sources, checkpoints each to `_fetch_cache/`, builds the FAISS index, and atomically swaps it into the `vect_store` Docker volume. It takes several minutes on first run.

5. The web UI is available at `https://<your_domain>`.

## Updating the index

Re-run `build_index.py` inside the container whenever you want to refresh the documentation:

```bash
# Re-fetch all sources and rebuild
docker exec rag-docs-chameleon-rag-app-1 python build_index.py

# Skip fetching, embed from existing cache (useful if the embedding API was down during a prior run)
docker exec rag-docs-chameleon-rag-app-1 python build_index.py --use-cache

# Re-fetch only specific sources, use cache for the rest
docker exec rag-docs-chameleon-rag-app-1 python build_index.py --refresh blog forum

docker restart rag-docs-chameleon-rag-app-1
```

## Local development

To run outside Docker, install dependencies and set up your `.env`:

```bash
pip install -r requirements.txt
```

Build the index:

```bash
python build_index.py
```

Run the web UI:

```bash
streamlit run web_rag.py
```

Or use the CLI:

```bash
python rag.py
```

## Evaluation

A golden question set and scoring pipeline are in `eval/`. To run an evaluation:

```bash
# Inside the container
docker exec rag-docs-chameleon-rag-app-1 python eval/run_pipeline.py

# Copy results out
docker cp rag-docs-chameleon-rag-app-1:/app/eval/results/ eval/results/

# Score locally (uses the eval venv)
eval/.venv/bin/python eval/score.py eval/results/run_<timestamp>.json --csv eval/results/scores.csv
```

Scoring uses cosine similarity between E5-Mistral embeddings of the generated and ground-truth answers. Results are appended to the CSV for comparison across runs.

## Documentation sources

| Source | Type |
|---|---|
| [chameleoncloud.readthedocs.io](https://chameleoncloud.readthedocs.io/en/latest/) | Primary docs |
| [python-chi.readthedocs.io](https://python-chi.readthedocs.io/en/latest/) | Python API docs |
| [chameleoncloud.gitbook.io/chi-edge](https://chameleoncloud.gitbook.io/chi-edge) | CHI@Edge docs |
| [chameleoncloud.gitbook.io/trovi](https://chameleoncloud.gitbook.io/trovi) | Trovi artifact docs |
| [blog.chameleoncloud.org](https://blog.chameleoncloud.org) | Tips, changelogs, featured posts |
| [forum.chameleoncloud.org](https://forum.chameleoncloud.org) | Community Q&A |
| [chameleoncloud.org](https://chameleoncloud.org) | FAQ, hardware, about pages |
