"""
Build (or refresh) the FAISS vector index from Chameleon docs.

Each documentation source is fetched independently and checkpointed to
_fetch_cache/<source>.json before anything is embedded. If a source fails,
the others are still saved and the existing live index is untouched.

The new index is written to a temporary directory and moved into place
only after the full build succeeds.

Usage:
    python build_index.py                  # re-fetch all sources, rebuild index
    python build_index.py --use-cache      # skip fetch, embed from existing cache
    python build_index.py --refresh blog   # re-fetch only blog, use cache for rest
"""
import argparse
import json
import os
import shutil

from langchain_core.documents import Document

from loader import SOURCES
from rag import create_vectorstore, VECT_STORE_PATH

CACHE_DIR = "_fetch_cache"
TEMP_STORE = VECT_STORE_PATH + "_building"


def save_cache(name, docs):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump([{"content": d.page_content, "metadata": d.metadata} for d in docs], f)
    os.replace(tmp, path)


def load_cache(name):
    path = os.path.join(CACHE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]


def fetch_source(name, fetch_fn):
    print(f"  Fetching {name}...")
    try:
        docs = fetch_fn()
        save_cache(name, docs)
        print(f"    {name}: {len(docs)} documents (saved to cache)")
        return docs
    except Exception as e:
        print(f"    {name}: FAILED — {e}")
        cached = load_cache(name)
        if cached:
            print(f"    {name}: falling back to cached {len(cached)} documents")
            return cached
        print(f"    {name}: no cache available, skipping")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-cache", action="store_true",
        help="Skip fetching entirely, build index from existing cache files",
    )
    group.add_argument(
        "--refresh", nargs="*", metavar="SOURCE",
        help="Re-fetch only these sources (default: all). Use with no args to refresh all.",
    )
    args = parser.parse_args()

    # Determine which sources to re-fetch
    if args.use_cache:
        refresh_set = set()
    elif args.refresh is None:
        refresh_set = set(SOURCES)       # default: re-fetch everything
    else:
        refresh_set = set(args.refresh) if args.refresh else set(SOURCES)

    print("=== Fetch phase ===")
    all_docs = []
    for name, fetch_fn in SOURCES.items():
        if name in refresh_set:
            docs = fetch_source(name, fetch_fn)
        else:
            docs = load_cache(name) or []
            print(f"  {name}: {len(docs)} documents (from cache)")
        all_docs.extend(docs)

    print(f"\nTotal: {len(all_docs)} documents")

    if not all_docs:
        print("No documents loaded — aborting.")
        raise SystemExit(1)

    print("\n=== Embed phase ===")
    if os.path.exists(TEMP_STORE):
        shutil.rmtree(TEMP_STORE)
    create_vectorstore(all_docs, save_path=TEMP_STORE)

    print("\n=== Swap phase ===")
    # vect_store is a Docker volume mount point and cannot be renamed.
    # Move individual files in instead, overwriting the old index.
    os.makedirs(VECT_STORE_PATH, exist_ok=True)
    for fname in os.listdir(TEMP_STORE):
        shutil.move(os.path.join(TEMP_STORE, fname), os.path.join(VECT_STORE_PATH, fname))
    shutil.rmtree(TEMP_STORE)

    print(f"Index live at '{VECT_STORE_PATH}'. Done.")
