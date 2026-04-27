#!/usr/bin/env python3
"""Print a JSON feedback report to stdout.

Usage:
    python feedback_report.py
    python feedback_report.py --db /path/to/feedback.db
    python feedback_report.py --pretty
"""
import argparse
import json

from feedback_store import FeedbackStore, FEEDBACK_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Generate a feedback report as JSON.")
    parser.add_argument(
        "--db",
        default=FEEDBACK_DB_PATH,
        help=f"Path to the SQLite database (default: {FEEDBACK_DB_PATH})",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output",
    )
    args = parser.parse_args()

    store = FeedbackStore(db_path=args.db)
    report = store.get_report()
    print(json.dumps(report, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
