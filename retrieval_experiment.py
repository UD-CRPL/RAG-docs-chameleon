"""
Retrieval configuration experiment.

Holds the reranker and threshold fixed; varies only retrieval parameters
to measure how each configuration affects pool quality and coverage.

Metrics per question:
  pool_top1       — highest reranker score in the candidate pool
  pool_mean       — mean reranker score across all candidates
  selected_count  — chunks that pass the 0.5 threshold (context coverage)
  docs_fraction   — fraction of selected chunks from readthedocs (balance)

Run inside the container:
    docker exec rag-docs-chameleon-rag-app-1 python retrieval_experiment.py
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/app")
from rag import get_embeddings_model, get_reranker, load_parents, VECT_STORE_PATH, MIN_RERANKER_SCORE
from langchain_community.vectorstores import FAISS

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "eval", "golden_set.json")
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "eval", "results")
QUERY_PREFIX    = "A question regarding the Chameleon Cloud testbed: "

CONFIGS = [
    {
        "name":        "baseline",
        "description": "Current config: per-type balancing (top 10/type), fetch_k=50, query prefix",
        "fetch_k":     50,
        "balance":     True,
        "per_type_k":  10,
        "prefix":      True,
    },
    {
        "name":        "A_no_balance",
        "description": "No balancing, fetch_k=50, query prefix",
        "fetch_k":     50,
        "balance":     False,
        "per_type_k":  None,
        "prefix":      True,
    },
    {
        "name":        "B_fetch100",
        "description": "No balancing, fetch_k=100, query prefix",
        "fetch_k":     100,
        "balance":     False,
        "per_type_k":  None,
        "prefix":      True,
    },
    {
        "name":        "C_fetch200",
        "description": "No balancing, fetch_k=200, query prefix",
        "fetch_k":     200,
        "balance":     False,
        "per_type_k":  None,
        "prefix":      True,
    },
    {
        "name":        "D_no_prefix",
        "description": "No balancing, fetch_k=50, NO query prefix",
        "fetch_k":     50,
        "balance":     False,
        "per_type_k":  None,
        "prefix":      False,
    },
]


def retrieve(question, vectorstore, reranker, cfg, k=6):
    """Run one retrieval configuration and return scored candidates."""
    query = (QUERY_PREFIX + question) if cfg["prefix"] else question

    # Similarity search
    all_scored = vectorstore.similarity_search_with_relevance_scores(query, k=cfg["fetch_k"])

    # Optional per-type balancing
    if cfg["balance"]:
        by_type = defaultdict(list)
        for doc, _ in all_scored:
            src = doc.metadata.get("source_type", "other")
            by_type[src].append(doc)
        candidates = []
        for docs in by_type.values():
            candidates.extend(docs[:cfg["per_type_k"]])
    else:
        candidates = [doc for doc, _ in all_scored]

    # Reranker scoring
    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # URL dedup + threshold selection
    seen_urls = set()
    selected  = []
    for doc, score in ranked:
        if score < MIN_RERANKER_SCORE:
            break
        url = doc.metadata.get("source", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        selected.append((doc, score))
        if len(selected) >= k:
            break

    return ranked, selected


def metrics(ranked, selected):
    """Compute the four summary metrics from a retrieval result."""
    all_scores = [float(s) for _, s in ranked]
    pool_top1  = max(all_scores) if all_scores else 0.0
    pool_mean  = sum(all_scores) / len(all_scores) if all_scores else 0.0

    selected_count = len(selected)
    if selected_count:
        docs_selected = sum(
            1 for doc, _ in selected
            if "readthedocs" in doc.metadata.get("source", "")
            or "python-chi" in doc.metadata.get("source", "")
        )
        docs_fraction = docs_selected / selected_count
    else:
        docs_fraction = 0.0

    return {
        "pool_top1":      round(pool_top1, 4),
        "pool_mean":      round(pool_mean, 4),
        "selected_count": selected_count,
        "docs_fraction":  round(docs_fraction, 3),
    }


def print_summary(all_results):
    """Print a compact comparison table aggregated across all questions."""
    cfg_names = list(CONFIGS[0].keys())  # not used directly — iterate CONFIGS

    # Aggregate per config
    agg = {cfg["name"]: defaultdict(list) for cfg in CONFIGS}
    for qresult in all_results:
        for cfg_name, m in qresult["configs"].items():
            for k, v in m.items():
                agg[cfg_name][k].append(v)

    metric_keys = ["pool_top1", "pool_mean", "selected_count", "docs_fraction"]
    col_w = 14

    header = f"{'Config':<20}" + "".join(f"{k:>{col_w}}" for k in metric_keys)
    print("\n" + "─" * len(header))
    print("AGGREGATE RESULTS  (mean across all questions)")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for cfg in CONFIGS:
        name = cfg["name"]
        row  = f"{name:<20}"
        for k in metric_keys:
            vals = agg[name][k]
            mean = sum(vals) / len(vals) if vals else 0
            row += f"{mean:>{col_w}.3f}"
        print(row)
    print("─" * len(header))

    # Per-question breakdown for pool_top1 (most diagnostic metric)
    print("\nPER-QUESTION  pool_top1  (did we find a good chunk?)")
    print(f"{'Q':<4} {'Category':<18}" + "".join(f"{cfg['name']:>{col_w}}" for cfg in CONFIGS))
    print("─" * (22 + col_w * len(CONFIGS)))
    for qresult in all_results:
        row = f"{qresult['id']:<4} {qresult['category']:<18}"
        for cfg in CONFIGS:
            val = qresult["configs"][cfg["name"]]["pool_top1"]
            row += f"{val:>{col_w}.3f}"
        print(row)


def main():
    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    print("Loading models...")
    vectorstore = FAISS.load_local(
        VECT_STORE_PATH, get_embeddings_model(), allow_dangerous_deserialization=True
    )
    reranker = get_reranker()
    print(f"Ready. Running {len(golden)} questions × {len(CONFIGS)} configs...\n")

    all_results = []
    for entry in golden:
        qid      = entry["id"]
        question = entry["question"]
        print(f"  Q{qid:02d}: {question[:60]}...")

        cfg_results = {}
        for cfg in CONFIGS:
            t0 = time.time()
            ranked, selected = retrieve(question, vectorstore, reranker, cfg)
            m = metrics(ranked, selected)
            m["latency_s"] = round(time.time() - t0, 2)
            cfg_results[cfg["name"]] = m
            print(f"         {cfg['name']:<20} top1={m['pool_top1']:.3f}  "
                  f"mean={m['pool_mean']:.3f}  kept={m['selected_count']}  "
                  f"docs={m['docs_fraction']:.0%}  ({m['latency_s']}s)")

        all_results.append({
            "id":       qid,
            "category": entry["category"],
            "question": question,
            "configs":  cfg_results,
        })

    print_summary(all_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"retrieval_experiment_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "threshold": MIN_RERANKER_SCORE,
            "configs":   CONFIGS,
            "results":   all_results,
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
