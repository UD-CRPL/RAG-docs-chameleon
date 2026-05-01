"""
Run the RAG pipeline against the golden question set and save results for manual review.

Captures retrieval diagnostics (reranker scores, source breakdown, threshold behaviour)
alongside the model's answer. No automatic scoring — answers are for human assessment.

Run inside the container:
    docker exec rag-docs-chameleon-rag-app-1 python eval/run_pipeline.py

Copy results out:
    docker cp rag-docs-chameleon-rag-app-1:/app/eval/results/<run_file>.json eval/results/
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/app")
from rag import load_vectorstore, load_parents, create_llm_chain, build_context, VECT_STORE_PATH, MIN_RERANKER_SCORE

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results")


def retrieval_stats(debug_candidates):
    """Summarise reranker scores and source breakdown from debug_candidates."""
    by_type = defaultdict(list)
    for c in debug_candidates:
        src = "readthedocs" if "readthedocs" in c["url"] else \
              "blog"        if "blog.chameleon" in c["url"] else \
              "other"
        by_type[src].append(c["score"])

    stats = {
        "total_candidates": len(debug_candidates),
        "passed_threshold": sum(1 for c in debug_candidates if c["selected"]),
        "threshold_used":   MIN_RERANKER_SCORE,
        "by_source": {},
    }
    for src, scores in by_type.items():
        stats["by_source"][src] = {
            "n":         len(scores),
            "avg_score": round(sum(scores) / len(scores), 3),
            "top_score": round(max(scores), 3),
        }
    return stats


def main():
    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    print("Loading vector store and LLM chain...")
    vectorstore = load_vectorstore(VECT_STORE_PATH)
    parents     = load_parents(VECT_STORE_PATH)
    chain       = create_llm_chain()
    print(f"Ready. Running {len(golden)} questions...\n")

    results = []
    for entry in golden:
        qid      = entry["id"]
        question = entry["question"]
        print(f"  [{qid:02d}/{len(golden)}] {question[:70]}...")
        t0 = time.time()
        try:
            sources, context, debug_candidates = build_context(question, vectorstore, parents)
            response = chain.invoke({"question": question, "context": context, "history": []})
            answer   = response.content
            stats    = retrieval_stats(debug_candidates)
            status   = "ok"
        except Exception as e:
            sources, context, debug_candidates = [], "", []
            answer, stats, status = "", {}, f"error: {e}"
        elapsed = round(time.time() - t0, 2)

        results.append({
            "id":               qid,
            "category":         entry["category"],
            "question":         question,
            "answer":           answer,
            "sources":          sources,
            "retrieval":        stats,
            "latency_s":        elapsed,
            "status":           status,
        })

        chunks_kept = stats.get("passed_threshold", 0)
        breakdown   = "  ".join(
            f"{t}:{v['n']}(top {v['top_score']:.2f})"
            for t, v in stats.get("by_source", {}).items()
        )
        print(f"         {elapsed:.1f}s  kept={chunks_kept}/{stats.get('total_candidates', 0)}  {breakdown}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump({"timestamp": timestamp, "threshold": MIN_RERANKER_SCORE, "results": results}, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone. {ok}/{len(results)} succeeded. Results saved to {output_path}")


if __name__ == "__main__":
    main()
