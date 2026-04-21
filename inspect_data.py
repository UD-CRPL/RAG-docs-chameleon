"""
Inspect the contents of the indexed pages.json to assess data quality.

Run inside the container:
    docker exec rag-docs-chameleon-rag-app-1 python inspect_data.py

Or locally if vect_store/ is accessible:
    python inspect_data.py
"""
import json
import statistics
from collections import defaultdict
from rag import VECT_STORE_PATH, load_pages
from loader import _source_type

SHORT_PAGE_THRESHOLD = 300   # chars — likely nav-only or failed scrape
LONG_PAGE_THRESHOLD  = 50_000  # chars — may contain a lot of noise


def summarize():
    pages = load_pages(VECT_STORE_PATH)

    by_type: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for url, content in pages.items():
        stype = _source_type(url)
        by_type[stype].append((url, len(content)))

    all_lengths = [length for entries in by_type.values() for _, length in entries]

    print("=" * 65)
    print(f"TOTAL INDEXED PAGES: {len(pages)}")
    print(f"Total content: {sum(all_lengths) / 1_000:.0f}k chars")
    print("=" * 65)

    # ── Per-type breakdown ──────────────────────────────────────────
    print("\nBREAKDOWN BY SOURCE TYPE")
    print(f"{'Type':<18} {'Pages':>6}  {'Min':>7}  {'Median':>7}  {'Max':>7}  {'Total':>9}")
    print("-" * 65)
    for stype, entries in sorted(by_type.items(), key=lambda x: -len(x[1])):
        lengths = [l for _, l in entries]
        print(
            f"{stype:<18} {len(lengths):>6}  "
            f"{min(lengths):>7,}  {statistics.median(lengths):>7,.0f}  "
            f"{max(lengths):>7,}  {sum(lengths) / 1000:>8.0f}k"
        )

    # ── Short pages ─────────────────────────────────────────────────
    short = [
        (url, len(content), _source_type(url))
        for url, content in pages.items()
        if len(content) < SHORT_PAGE_THRESHOLD
    ]
    short.sort(key=lambda x: x[1])

    print(f"\nSHORT PAGES (< {SHORT_PAGE_THRESHOLD} chars) — potential nav/empty pages: {len(short)}")
    if short:
        print("-" * 65)
        for url, length, stype in short[:20]:
            print(f"  [{stype}] {length:>5} chars  {url}")
            snippet = pages[url].replace("\n", " ").strip()[:120]
            print(f"           preview: {snippet!r}")
            print()

    # ── Long pages ──────────────────────────────────────────────────
    long = [
        (url, len(content), _source_type(url))
        for url, content in pages.items()
        if len(content) > LONG_PAGE_THRESHOLD
    ]
    long.sort(key=lambda x: -x[1])

    print(f"\nLONG PAGES (> {LONG_PAGE_THRESHOLD:,} chars) — potential noise: {len(long)}")
    if long:
        print("-" * 65)
        for url, length, stype in long[:10]:
            print(f"  [{stype}] {length:>8,} chars  {url}")

    # ── Per-type short-page rate ─────────────────────────────────────
    print("\nSHORT-PAGE RATE BY TYPE")
    print("-" * 65)
    for stype, entries in sorted(by_type.items(), key=lambda x: -len(x[1])):
        total = len(entries)
        n_short = sum(1 for _, l in entries if l < SHORT_PAGE_THRESHOLD)
        pct = 100 * n_short / total if total else 0
        bar = "█" * int(pct / 5)
        print(f"  {stype:<18} {n_short:>3}/{total:<4}  {pct:>5.1f}%  {bar}")

    print("\nDone.")


if __name__ == "__main__":
    summarize()
