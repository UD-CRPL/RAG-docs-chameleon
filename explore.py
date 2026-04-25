"""
Retrieval diagnostic tool — understand what the pipeline is doing for any question.

Usage (inside container):
    python explore.py "How do I reserve a bare metal node?"
    python explore.py  # interactive mode
"""
import sys
import os
os.environ['USER_AGENT'] = 'myagent'

from collections import defaultdict
from rag import get_embeddings_model, get_reranker, load_parents, VECT_STORE_PATH
from langchain_community.vectorstores import FAISS

FETCH_K   = 50
MMR_K     = 20
PER_TYPE_K = 10
SELECT_K  = 6

SOURCE_LABELS = {
    "readthedocs":   "Chameleon Docs",
    "python_chi":    "Python CHI Docs",
    "blog":          "Blog",
    "forum":         "Forum",
    "gitbook":       "CHI@Edge/Trovi",
    "chameleon_org": "Chameleon.org",
}


def label(src_type):
    return SOURCE_LABELS.get(src_type, src_type or "other")


def summarise(scores):
    if not scores:
        return "n=0"
    return f"n={len(scores):2d}  avg={sum(scores)/len(scores):.3f}  top={max(scores):.3f}  bot={min(scores):.3f}"


def explore(question, vectorstore, parents, reranker):
    query = f"A question regarding the Chameleon Cloud testbed: {question}"

    # ── Stage 1: raw similarity search ───────────────────────────────────────
    all_scored = vectorstore.similarity_search_with_relevance_scores(query, k=FETCH_K)

    by_type_sim = defaultdict(list)
    for doc, score in all_scored:
        src = doc.metadata.get('source_type', 'other')
        by_type_sim[src].append(score)

    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")

    print(f"\n── Stage 1: Raw similarity  (top {FETCH_K} chunks) ──────────────")
    for src_type, scores in sorted(by_type_sim.items()):
        print(f"  {label(src_type):<20} {summarise(scores)}")

    # ── Stage 2: per-type balanced pool ──────────────────────────────────────
    by_type_docs = defaultdict(list)
    for doc, score in all_scored:
        src = doc.metadata.get('source_type', 'other')
        by_type_docs[src].append((doc, score))

    pool = []
    for src_type, items in by_type_docs.items():
        pool.extend(items[:PER_TYPE_K])
    pool_docs = [doc for doc, _ in pool]

    print(f"\n── Stage 2: Reranker pool   (top {PER_TYPE_K}/type) ──────────────")
    pool_by_type = defaultdict(list)
    for doc, sim in pool:
        src = doc.metadata.get('source_type', 'other')
        pool_by_type[src].append(sim)
    for src_type, scores in sorted(pool_by_type.items()):
        print(f"  {label(src_type):<20} {summarise(scores)}")

    # ── Stage 3: reranker scores ──────────────────────────────────────────────
    pairs  = [(query, doc.page_content) for doc in pool_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(pool_docs, scores), key=lambda x: x[1], reverse=True)

    by_type_rerank = defaultdict(list)
    for doc, sc in ranked:
        src = doc.metadata.get('source_type', 'other')
        by_type_rerank[src].append(sc)

    print(f"\n── Stage 3: Reranker scores (all {len(ranked)} candidates) ──────────")
    for src_type, sc_list in sorted(by_type_rerank.items()):
        print(f"  {label(src_type):<20} {summarise(sc_list)}")

    # ── Stage 4: final selection ──────────────────────────────────────────────
    seen_urls = set()
    selected = []
    selected_by_type = defaultdict(int)
    for doc, sc in ranked:
        url = doc.metadata.get('source', '')
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        if len(selected) < SELECT_K:
            selected.append((doc, sc))
            selected_by_type[doc.metadata.get('source_type', 'other')] += 1

    print(f"\n── Stage 4: Selected ({SELECT_K}) ──────────────────────────────────")
    for src_type, count in sorted(selected_by_type.items()):
        print(f"  {label(src_type):<20} {count} chunk(s)")
    print()
    for i, (doc, sc) in enumerate(selected, 1):
        src = doc.metadata.get('source_type', 'other')
        url = doc.metadata.get('source', '')
        parent = parents.get(doc.metadata.get('parent_id', ''), {})
        title  = parent.get('content', '')[:60].replace('\n', ' ').strip()
        print(f"  {i}. [{sc:+.2f}] {label(src):18}  {url}")
        print(f"        \"{title}...\"")

    # ── Threshold simulation ──────────────────────────────────────────────────
    print(f"\n── Threshold simulation ─────────────────────────────────────────")
    print(f"  {'Threshold':<12} {'Chunks kept':<14} {'Breakdown'}")
    for thresh in [0.05, 0.10, 0.20, 0.30, 0.50]:
        seen = set()
        kept = []
        for doc, sc in ranked:
            if sc < thresh:
                continue
            url = doc.metadata.get('source', '')
            if url and url in seen:
                continue
            seen.add(url)
            kept.append((doc, sc))
            if len(kept) >= SELECT_K:
                break
        by_t = defaultdict(int)
        for doc, _ in kept:
            by_t[doc.metadata.get('source_type', 'other')] += 1
        breakdown = "  ".join(f"{label(t)} ×{n}" for t, n in sorted(by_t.items()))
        print(f"  {thresh:<12.2f} {len(kept):<14} {breakdown or '(none)'}")


def main():
    print("Loading models (this takes a moment on first run)...")
    vectorstore = FAISS.load_local(
        VECT_STORE_PATH, get_embeddings_model(), allow_dangerous_deserialization=True
    )
    parents  = load_parents()
    reranker = get_reranker()
    print("Ready.\n")

    questions = sys.argv[1:] if len(sys.argv) > 1 else []

    if questions:
        for q in questions:
            explore(q, vectorstore, parents, reranker)
    else:
        print("Interactive mode — enter questions, blank line to quit.")
        while True:
            try:
                q = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            explore(q, vectorstore, parents, reranker)


if __name__ == "__main__":
    main()
