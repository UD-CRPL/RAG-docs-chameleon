"""
Score a pipeline run against the golden set using embedding-based semantic similarity.

Run locally after copying results out of the container:
    python eval/score.py eval/results/run_YYYYMMDD_HHMMSS.json

Optionally append scores to a CSV for comparison with prior runs:
    python eval/score.py eval/results/run_YYYYMMDD_HHMMSS.json --csv eval/results/scores.csv

Similarity method: cosine similarity between BAAI/bge-large-en-v1.5 embeddings of the
generated answer and the ground truth answer.
"""
import argparse
import csv
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def get_embeddings(texts: list[str]) -> np.ndarray:
    return _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_run(run_path: str, csv_path: str | None = None):
    with open(run_path) as f:
        run = json.load(f)

    results   = run["results"]
    timestamp = run["timestamp"]

    print(f"Scoring {len(results)} results from {timestamp}...\n")

    # Batch embed all ground truths and generated answers in two calls
    ground_truths     = [r["ground_truth"]     for r in results]
    generated_answers = [r["generated_answer"] for r in results]

    print("  Embedding ground truths...")
    gt_vecs  = get_embeddings(ground_truths)
    print("  Embedding generated answers...")
    gen_vecs = get_embeddings(generated_answers)

    scored = []
    for i, r in enumerate(results):
        sim = cosine_sim(gt_vecs[i], gen_vecs[i]) if r["status"] == "ok" else None
        scored.append({**r, "semantic_similarity": round(sim, 4) if sim is not None else None})

    # ── Print per-question results ─────────────────────────────────
    print(f"\n{'ID':<4} {'Category':<18} {'Sim':>6}  {'Latency':>8}  Question")
    print("-" * 80)
    for r in scored:
        sim_str = f"{r['semantic_similarity']:.4f}" if r["semantic_similarity"] is not None else "  ERR "
        q_short = r["question"][:52]
        print(f"{r['id']:<4} {r['category']:<18} {sim_str:>6}  {r['latency_s']:>6.1f}s  {q_short}")

    valid_sims = [r["semantic_similarity"] for r in scored if r["semantic_similarity"] is not None]

    # ── Summary by category ────────────────────────────────────────
    from collections import defaultdict
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        if r["semantic_similarity"] is not None:
            by_cat[r["category"]].append(r["semantic_similarity"])

    print(f"\nSUMMARY BY CATEGORY")
    print(f"{'Category':<20} {'N':>3}  {'Mean':>6}  {'Min':>6}  {'Max':>6}")
    print("-" * 50)
    for cat, sims in sorted(by_cat.items()):
        print(f"{cat:<20} {len(sims):>3}  {sum(sims)/len(sims):>6.4f}  {min(sims):>6.4f}  {max(sims):>6.4f}")

    overall = sum(valid_sims) / len(valid_sims) if valid_sims else 0
    print(f"\nOVERALL  n={len(valid_sims)}  mean={overall:.4f}  "
          f"min={min(valid_sims):.4f}  max={max(valid_sims):.4f}")

    # ── Optional CSV append ────────────────────────────────────────
    if csv_path:
        fieldnames = ["run_timestamp", "id", "category", "question",
                      "semantic_similarity", "latency_s", "status", "retrieved_sources"]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in scored:
                writer.writerow({
                    **r,
                    "run_timestamp":    timestamp,
                    "retrieved_sources": "|".join(r.get("retrieved_sources", [])),
                })
        print(f"\nScores appended to {csv_path}")

    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", help="Path to a run_TIMESTAMP.json from run_pipeline.py")
    parser.add_argument("--csv", help="Optional path to append scores CSV", default=None)
    args = parser.parse_args()
    score_run(args.run_file, args.csv)
