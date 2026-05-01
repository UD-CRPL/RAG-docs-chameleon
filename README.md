# Chameleon Docs Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the [Chameleon Cloud](https://chameleoncloud.org) testbed. It indexes documentation from multiple official sources and serves a Streamlit web UI backed by a Llama 3.3 70B language model via the [Tejas AI](https://ai.tejas.tacc.utexas.edu) API.

## Architecture

**Indexing** — `build_index.py` crawls documentation from ReadTheDocs, GitBook, the Chameleon blog, the community forum, and chameleoncloud.org. Each source is fetched independently and checkpointed to `_fetch_cache/` before anything is embedded, so a single source failure doesn't abort the run. Documents are split into large parent chunks (~2000 chars) for context and small child chunks (~400 chars) for embedding. Each child chunk is prefixed with a `[source_type: page title]` header so the embedding captures document-level context. Child chunks are indexed in a FAISS vector store; parent content is stored in `parents.json`. The new index is built in a temporary directory and atomically swapped into place only after the full build succeeds.

**Retrieval** — Incoming questions are embedded with `BAAI/bge-large-en-v1.5` and matched against child chunks via similarity search. Candidates are drawn in a per-source-type balanced pool (to prevent any one source from dominating before scoring), then re-ranked by a cross-encoder (`BAAI/bge-reranker-base`). Chunks below a minimum reranker score are discarded. The corresponding parent chunks are assembled into a tiered context block: official docs (readthedocs + python-chi) are placed in a PRIMARY DOCUMENTATION section; blog posts, forum threads, and other sources go into a SUPPLEMENTARY CONTEXT section. The LLM is instructed to ground answers in primary documentation first.

**Generation** — Meta-Llama-3.3-70B-Instruct generates answers strictly from the retrieved context. Conversation history (last 3 turns) is included for multi-turn use.

**Feedback & logging** — Every query and answer is appended to `session_log.jsonl`. Users can rate responses thumbs-up/down in the UI; negative ratings prompt a failure-category selector and an optional comment. Feedback is stored in a SQLite database (`feedback.db`) and can be summarised with `feedback_report.py`.

**Deployment** — Docker Compose runs the Streamlit app behind a Traefik reverse proxy with automatic TLS. Named volumes persist the vector store, fetch cache, feedback database, and session log across container restarts.

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

The chatbot answers from a static snapshot of the documentation that was current at index-build time. As Chameleon's documentation evolves, the index must be rebuilt to reflect those changes.

**When to re-index:**
- After a significant Chameleon release or infrastructure change (new hardware sites, new features)
- When users report answers that seem outdated — cross-check with the current docs
- Periodically (monthly is a reasonable baseline) to pick up incremental doc edits
- After editing `loader.py` to add, remove, or tune a source

Re-run `build_index.py` inside the container:

```bash
# Re-fetch all sources and rebuild (preferred — picks up any upstream changes)
docker exec rag-docs-chameleon-rag-app-1 python build_index.py

# Skip fetching, embed from existing cache (use if the network was unreliable during a prior run)
docker exec rag-docs-chameleon-rag-app-1 python build_index.py --use-cache

# Re-fetch only specific sources, use cache for the rest
docker exec rag-docs-chameleon-rag-app-1 python build_index.py --refresh blog forum

docker restart rag-docs-chameleon-rag-app-1
```

### Active documentation sources

The `SOURCES` dict in `loader.py` controls which sources are fetched and indexed. Currently active:

| Key | Content |
|---|---|
| `readthedocs` | chameleoncloud.readthedocs.io + python-chi.readthedocs.io |
| `blog` | blog.chameleoncloud.org — tips-and-tricks category only |

Additional fetch functions exist in `loader.py` for forum, GitBook, and chameleoncloud.org pages but are not currently wired into `SOURCES`. They can be re-enabled by adding them back to the dict.

### Adding a new documentation source

1. Add a fetch function to `loader.py` that returns a list of `Document` objects, each with `page_content`, `source`, `source_type`, and `title` in its metadata.
2. Add an entry to the `SOURCES` dict:
   ```python
   SOURCES = {
       "readthedocs": fetch_readthedocs,
       "blog": fetch_blog,
       "my_new_source": fetch_my_new_source,  # add here
   }
   ```
3. Rebuild the index: `python build_index.py --refresh my_new_source`

The new source will be classified as `SUPPLEMENTARY CONTEXT` by default. To promote it to `PRIMARY DOCUMENTATION` (official docs treated as authoritative by the model), add its `source_type` to the `PRIMARY_TYPES` set in `rag.py`.

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

Scoring uses cosine similarity between BGE embeddings of the generated and ground-truth answers. Results are appended to the CSV for comparison across runs.

## Maintenance

Keeping the chatbot reliable requires attention in three areas: index freshness, service health, and dependency hygiene.

**Index freshness** is the most routine task. The index is a static snapshot; the chatbot cannot know when upstream documentation changes. Until automated change-detection is implemented, re-indexing must be triggered manually (see [Updating the index](#updating-the-index)). Feedback data and session logs are the primary signals that something is stale — a cluster of negative ratings on a specific topic, or a pattern of "I don't know" answers where docs exist, both indicate the index needs refreshing.

**Monitoring the running service:**
- `session_log.jsonl` — every query, the sources retrieved, full context sent to the model, the answer, and latency. Useful for spotting retrieval failures and latency regressions.
- `feedback.db` — user ratings and failure-category labels. Run `python feedback_report.py --pretty` to see a summary. High rates of "Inaccurate" or "Outdated" ratings point to specific documentation gaps or a stale index.
- The debug UI (enable with `SHOW_DEBUG = True` in `web_rag.py`) shows reranker scores and candidate counts per response, useful for diagnosing individual retrieval failures.

**Dependency and model hygiene:**
- The embedding model (`BAAI/bge-large-en-v1.5`) and reranker (`BAAI/bge-reranker-base`) are pinned as Python dependencies. Update them deliberately — changing models requires a full index rebuild because stored embeddings are model-specific.
- The LLM (Meta-Llama-3.3-70B-Instruct via Tejas AI) is referenced by name in `rag.py`. Check the Tejas AI API for model version updates or deprecations.
- Review `requirements.txt` for dependency updates periodically, especially `langchain`, `sentence-transformers`, and `faiss-cpu`.

**Known documentation blind spots** that the index cannot currently answer well:
- CHI-in-a-Box deployment and administration
- FPGA vs GPU node comparison and selection guidance
- Acceptable use policy (e.g., cryptocurrency mining)
- Detailed per-site hardware specifications

These are best addressed by writing targeted documentation stubs and adding them to the primary index, rather than relying on blog posts.

## Roadmap

The full roadmap is in [ROADMAP.md](ROADMAP.md). Key planned improvements:

**Retrieval quality**
- **Two-store architecture** — split the FAISS index into a primary store (readthedocs, python-chi) and a supplemental store (curated blog posts, Trovi). This eliminates the need for heuristic source-type balancing and gives cleaner control over what the model treats as authoritative.
- **Query expansion / HyDE** — embed a hypothetical answer instead of the raw question to improve recall for vocabulary-mismatched queries (e.g., "Error 403 Forbidden" vs the docs' phrasing "source your openrc file").
- **Per-source-type chunk sizing** — blog posts and reference docs have different structure; tuning chunk sizes per source type may improve retrieval precision.

**Content expansion**
- **Trovi artifact integration** — index Trovi artifact metadata and surface relevant artifacts as a companion UI element ("see it in action") without injecting them into model context.
- **Documentation gap-filling** — write short, focused stubs for topics with no current readthedocs coverage (CHI-in-a-Box, hardware comparisons, acceptable use) and add them to the primary store.
- **Re-enable forum and GitBook sources** — these were removed because they degraded retrieval quality in the shared pool; the two-store architecture above would make them safe to re-include.

**Live data integration**
- **Chameleon API MCP server** — a Model Context Protocol server wrapping the Chameleon REST APIs (Blazar reservations, node availability, hardware discovery) to answer dynamic questions like "what GPU nodes are available right now?" This is architecturally distinct from RAG — tool use rather than document retrieval — and requires designing an authentication flow.

**Operational automation**
- **Automated index freshness detection** — detect upstream documentation changes and trigger incremental re-indexing without manual intervention.
- **Expanded evaluation set** — grow the golden question set to 100+ questions with better category balance; add LLM-as-judge scoring for factual accuracy alongside the current retrieval diagnostics.

## Diagnostics

**Retrieval explorer** — `explore.py` walks through every stage of the pipeline for a single question and prints per-source statistics, reranker scores, and which chunks were selected:

```bash
# Inside the container
docker exec rag-docs-chameleon-rag-app-1 python explore.py "How do I reserve a bare metal node?"

# Or interactive mode
docker exec -it rag-docs-chameleon-rag-app-1 python explore.py
```

**Feedback report** — print a JSON summary of all user feedback collected so far:

```bash
docker exec rag-docs-chameleon-rag-app-1 python feedback_report.py --pretty
```

**Session log** — raw Q&A records (question, sources, context, answer, latency) are appended to `session_log.jsonl` and persisted in the `session_log` Docker volume.

**Debug UI** — set `SHOW_DEBUG = True` in `web_rag.py` to display a collapsible retrieval-debug panel below each answer in the web UI, showing every candidate chunk with its reranker score and selection status.

## Documentation sources

Sources currently active in the index:

| Source | Type |
|---|---|
| [chameleoncloud.readthedocs.io](https://chameleoncloud.readthedocs.io/en/latest/) | Primary docs |
| [python-chi.readthedocs.io](https://python-chi.readthedocs.io/en/latest/) | Python API docs |
| [blog.chameleoncloud.org](https://blog.chameleoncloud.org) | Tips-and-tricks posts |

Sources with fetch support in `loader.py` but not currently indexed:

| Source | Type | Notes |
|---|---|---|
| [chameleoncloud.gitbook.io/chi-edge](https://chameleoncloud.gitbook.io/chi-edge) | CHI@Edge docs | Degraded retrieval quality in shared pool |
| [chameleoncloud.gitbook.io/trovi](https://chameleoncloud.gitbook.io/trovi) | Trovi artifact docs | Planned for dedicated supplemental store |
| [forum.chameleoncloud.org](https://forum.chameleoncloud.org) | Community Q&A | Degraded retrieval quality in shared pool |
| [chameleoncloud.org](https://chameleoncloud.org) | FAQ, hardware, about pages | Low content density relative to readthedocs |
