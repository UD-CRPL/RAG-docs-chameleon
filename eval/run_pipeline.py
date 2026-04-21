"""
Run the current RAG pipeline against the golden question set and save outputs.

Must be run inside the container (needs access to the vector store volume):
    docker cp eval/run_pipeline.py rag-docs-chameleon-rag-app-1:/app/eval/run_pipeline.py
    docker exec rag-docs-chameleon-rag-app-1 python eval/run_pipeline.py

Then copy results out:
    docker cp rag-docs-chameleon-rag-app-1:/app/eval/results/<run_file>.json eval/results/
"""
import json
import os
import sys
import time
from datetime import datetime

# Run from /app inside the container
sys.path.insert(0, "/app")
from rag import load_vectorstore, create_llm_chain, load_pages, build_context, VECT_STORE_PATH

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results")


def run_question(question: str, retriever, chain, pages: dict) -> dict:
    sources, context = build_context(question, retriever, pages)
    response = chain.invoke({"question": question, "context": context, "history": []})
    return {
        "generated_answer": response.content,
        "retrieved_sources": sources,
    }


def main():
    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    print("Loading vector store and LLM chain...")
    retriever = load_vectorstore(VECT_STORE_PATH)
    chain     = create_llm_chain()
    pages     = load_pages(VECT_STORE_PATH)
    print(f"Ready. Running {len(golden)} questions...\n")

    results = []
    for entry in golden:
        qid      = entry["id"]
        question = entry["question"]
        print(f"  [{qid:02d}/{len(golden)}] {question[:70]}...")
        t0 = time.time()
        try:
            output = run_question(question, retriever, chain, pages)
            status = "ok"
        except Exception as e:
            output = {"generated_answer": "", "retrieved_sources": [], "num_chunks": 0}
            status = f"error: {e}"
        elapsed = time.time() - t0

        results.append({
            "id":               qid,
            "category":         entry["category"],
            "question":         question,
            "ground_truth":     entry["ground_truth"],
            "generated_answer": output["generated_answer"],
            "retrieved_sources": output["retrieved_sources"],
            "latency_s":        round(elapsed, 2),
            "status":           status,
        })
        print(f"         {elapsed:.1f}s  status={status}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone. {ok}/{len(results)} succeeded. Results saved to {output_path}")


if __name__ == "__main__":
    main()
