# Roadmap

## Completed

- **Parent document retrieval** — two-level chunking (400-char child chunks for FAISS, 2000-char parent chunks for context). Replaced earlier approach of returning entire pages as context.
- **Source diversity filter** — MMR retrieval + deduplication by parent ID. Primary sources (readthedocs, blog) fill context slots first; specialized sources (forum, gitbook, etc.) only included if slots remain, capped at one chunk per type. Replaced earlier flat `max_per_type` cap.
- **Contextual chunk headers** — each child chunk is prefixed with `[source_type: page title]` at index time so the embedding captures document-level context. Page titles extracted from `<h1>` tags (HTML), first `# heading` (Markdown), and topic title (forum). Requires reindex.
- **Robust indexing pipeline** — two-phase fetch→embed with per-source JSON checkpointing in `_fetch_cache/`. A source failure no longer aborts the run; prior cache is used as fallback. New index built to a temp directory and atomically swapped into place. `--use-cache` and `--refresh <source>` flags added.
- **Blog fetch retry** — exponential backoff on blog listing page requests to fix intermittent timeouts that were silently dropping all blog posts.
- **Evaluation pipeline** — golden set of 20 questions, semantic similarity scoring via E5-Mistral embeddings, CSV comparison across runs.

---

## In Progress

### Priority 2 — Cross-encoder reranker
Tried `FlashrankRerank` (ms-marco-MiniLM-L-12-v2) from `langchain_community`. Removed — the model was trained on web search passages and actively mis-ranked technical documentation, producing noticeably worse answers. May be worth revisiting with a domain-appropriate model, but deprioritized for now.

### Priority 3 — Hybrid search (BM25 + dense)
Dense embeddings miss exact keyword matches (CLI flags, function names, model names). BM25 is strong for these. Merge both rankings with Reciprocal Rank Fusion via LangChain's `EnsembleRetriever`. Requires adding `rank_bm25` and persisting the BM25 index alongside FAISS. **Requires reindex.**

### Priority 4 — Contextual chunk headers
Prepend each chunk with its source page title and section heading during indexing so the embedding captures document-level context. Improves retrieval precision for questions where the relevant chunk is ambiguous without its surrounding context. **Requires reindex.**

---

## Known Issues / Tech Debt

**Indexing**
- Forum indexing only fetches the `latest.json` endpoint (~30 topics). The full forum archive is not indexed.

**Retrieval**
- Parent chunk size (2000 chars) and child chunk size (400 chars) were chosen heuristically. No systematic tuning has been done. How-To and Troubleshooting categories are still below the original full-page baseline — chunk sizes may need adjustment per source type.
- No similarity score threshold — weak MMR matches still proceed to context assembly. Adding a minimum score cutoff could reduce noise.
- `max_per_type=2` diversity cap is a heuristic with no empirical basis.

**Evaluation**
- Semantic similarity (embedding cosine) is a proxy metric. It measures whether the generated answer is topically similar to the ground truth, but not faithfulness or correctness. LLM-as-judge scoring (e.g., rating factual accuracy, relevance, and completeness) would be more meaningful.
- Golden set is only 20 questions — too small for statistically reliable category-level comparisons. Should be expanded, especially for underrepresented categories (Negative Question has only 1 example).
- Eval run comparison is manual (CSV diff). No automated regression check that fails if mean score drops below a threshold.

---

## Future Improvements

- **Query expansion / HyDE** — generate a hypothetical answer to the question and embed that for retrieval, rather than embedding the raw question. Can improve recall for questions phrased differently from the documentation.
- **Per-source-type chunking** — forum posts and blog posts have different structure than technical docs. Tuning chunk sizes and separators per source type could improve indexing quality.
- **Index freshness** — detect when upstream documentation has changed and trigger incremental re-indexing, rather than requiring a full manual rebuild.
- **Answer caching** — cache responses to common questions to reduce latency and API cost.
- **Expanded golden set** — grow the evaluation set to 100+ questions with better category balance, including more edge cases (ambiguous questions, questions with no answer in the docs).
