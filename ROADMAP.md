# Roadmap

## Completed

- **Parent-child chunking** — 400-char child chunks for FAISS retrieval; 2000-char parent chunks assembled as model context. Retrieves on small chunks for precision, returns coherent context to the model.
- **Contextual chunk headers** — each child chunk prefixed with `[source_type: page title]` at index time so embeddings capture document-level context.
- **Robust indexing pipeline** — two-phase fetch→embed with per-source JSON checkpointing; atomic temp-dir swap; `--use-cache` and `--refresh <source>` flags.
- **Cross-encoder reranker** — replaced MMR-only retrieval with `BAAI/bge-reranker-base`. Fetches a per-type balanced candidate pool, reranks all candidates by (query, chunk) relevance, applies a 0.5 score threshold to cut weak matches.
- **Reranker score threshold** — chunks scoring below 0.5 are excluded from context. Model says "I don't know" rather than confabulating from weak matches.
- **Source-aware context assembly** — context split into PRIMARY DOCUMENTATION (readthedocs, python-chi) and SUPPLEMENTARY CONTEXT (blog) sections. System prompt instructs the model to ground answers in primary docs and use supplementary content only for examples and gap-filling.
- **Source filtering** — index narrowed to readthedocs + tips-and-tricks blog only. Removed changelogs, forum, gitbook, chameleon.org static pages. Root `index.html` and pages with <500 chars of content excluded.
- **Session logging** — every web app query logged to `session_log.jsonl` with question, sources, full context, answer, and latency.
- **Retrieval diagnostic tools** — `explore.py` (per-question stage-by-stage breakdown with threshold simulation) and `retrieval_experiment.py` (cross-config benchmark across golden set using reranker score as proxy metric).
- **Updated eval pipeline** — replaced semantic similarity scoring with retrieval diagnostics (reranker score stats, threshold coverage, source breakdown) + raw answers for manual quality assessment.
- **Debug panel in web UI** — per-response expander showing reranker score summary by source type, ✅/⬜/❌ candidate breakdown, and full context sent to model.

---

## Next Priorities

### 1 — Two-store architecture (docs + supplemental)
Split the single FAISS index into a primary store (readthedocs, python-chi) and a supplemental store (curated blog posts, Trovi artifacts). Search each store independently, combine candidates, rerank together. This eliminates the numerical dominance of blog chunks over docs chunks in the shared similarity pool without requiring heuristic balancing.

Supplemental store curation criterion: only content that covers topics the documentation doesn't address, or that demonstrates a documented concept in action. Not re-explanations of things the docs already cover (these introduce outdated or inconsistent information).

### 2 — Trovi artifact integration
Index Trovi artifact metadata (title, description, tags) as a separate lightweight store. After primary context is assembled, run a secondary lookup to surface a relevant artifact. Present artifacts as a companion UI element ("see it in action") rather than injecting them into model context — keeps the model's context clean while improving discoverability.

### 3 — Documentation gap-filling
Several recurring question types have no good answer in the current readthedocs corpus: CHI-in-a-Box overview, FPGA vs GPU comparison, acceptable use policy (crypto mining etc.), hardware site details. These gaps should be filled by writing short, focused documentation stubs added to the primary store — more durable and accurate than relying on blog posts.

### 4 — Live Chameleon API integration (MCP server)
A Model Context Protocol server wrapping the Chameleon REST APIs (Blazar reservations, hardware discovery, node availability). Lets the model answer dynamic questions like "what GPU nodes are available at CHI@UC right now?" by making live API calls with user credentials rather than retrieving from static docs. Architecturally distinct from the RAG pipeline — tool use rather than document retrieval. Requires thinking through the authentication flow.

---

## Known Issues

- **Latency** — reranker runs on CPU; fetch_k=100 takes ~13s per query. Experiment results (retrieval_experiment_20260425) show fetch_k=100 improves top1 score meaningfully vs fetch_k=50. GPU inference would make this practical.
- **Documentation blind spots** — CHI-in-a-Box, FPGA/GPU comparison, acceptable use policy answers are missing from the readthedocs corpus. Currently answered from blog posts with lower reliability.
- **`getting-started/index.html` over-matching** — this broad overview page appears in results for many unrelated questions. Consider excluding it from the index alongside the root `index.html`.
- **Q&A vocabulary mismatch** — some questions use phrasing that doesn't appear in the docs (e.g., "Error 403 Forbidden" vs "source your openrc file"). The reranker partially compensates but retrieval misses remain for vocabulary-distant queries.

---

## Future Improvements

- **Query expansion / HyDE** — embed a hypothetical answer rather than the raw question to improve recall for vocabulary-mismatched queries.
- **Per-source-type chunk sizing** — blog posts and docs have different structure; tuning chunk sizes per source type may improve retrieval quality.
- **Index freshness** — detect upstream documentation changes and trigger incremental re-indexing.
- **Expanded evaluation set** — grow golden set to 100+ questions with better category balance; add LLM-as-judge scoring for factual accuracy rather than relying solely on manual review.
