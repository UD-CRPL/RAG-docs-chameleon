import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

FEEDBACK_DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "/app/feedback.db")

FAILURE_CATEGORIES = [
    "Inaccurate",
    "Outdated",
    "Incomplete",
    "Didn't understand my question",
    "Too vague",
    "Missing context",
    "Other",
]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT    NOT NULL,
    session_id         TEXT    NOT NULL,
    question           TEXT    NOT NULL,
    response_hash      TEXT    NOT NULL,
    rating             TEXT    NOT NULL CHECK(rating IN ('positive','negative')),
    failure_categories TEXT    NOT NULL DEFAULT '[]',
    comment            TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_feedback_rating    ON feedback (rating);",
]


def hash_response(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


@dataclass
class FeedbackRecord:
    session_id: str
    question: str
    response_hash: str
    rating: str
    failure_categories: list = field(default_factory=list)
    comment: Optional[str] = None
    timestamp: Optional[str] = None


class FeedbackStore:
    """Persist and query user feedback records backed by SQLite.

    Each public method opens and closes its own connection so this class
    is safe to call from Streamlit's multi-threaded rerun model.
    """

    def __init__(self, db_path: str = FEEDBACK_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            for stmt in _CREATE_INDEXES:
                conn.execute(stmt)

    def save(self, record: FeedbackRecord) -> int:
        """Insert a feedback record and return the new row id."""
        record.timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback
                    (timestamp, session_id, question, response_hash,
                     rating, failure_categories, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.session_id,
                    record.question,
                    record.response_hash,
                    record.rating,
                    json.dumps(record.failure_categories),
                    record.comment,
                ),
            )
            return cur.lastrowid

    def already_rated(self, response_hash: str, session_id: str) -> bool:
        """Return True if this session has already submitted feedback for this response."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM feedback WHERE response_hash = ? AND session_id = ? LIMIT 1",
                (response_hash, session_id),
            ).fetchone()
            return row is not None

    def get_rated_hashes(self, session_id: str) -> set:
        """Return the set of response_hash values already rated in this session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT response_hash FROM feedback WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return {r["response_hash"] for r in rows}

    def get_report(self) -> dict:
        """Return aggregated feedback statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            pos = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = 'positive'"
            ).fetchone()[0]
            neg = total - pos

            def pct(n):
                return round(100 * n / total, 1) if total else 0.0

            cat_counts = {c: 0 for c in FAILURE_CATEGORIES}
            for row in conn.execute(
                "SELECT failure_categories FROM feedback WHERE rating = 'negative'"
            ).fetchall():
                for cat in json.loads(row["failure_categories"]):
                    if cat in cat_counts:
                        cat_counts[cat] += 1

            recent_rows = conn.execute(
                """
                SELECT timestamp, session_id, question, response_hash,
                       failure_categories, comment
                FROM feedback
                WHERE rating = 'negative'
                ORDER BY timestamp DESC
                LIMIT 10
                """
            ).fetchall()
            recent = [
                {
                    "timestamp": r["timestamp"],
                    "session_id": r["session_id"],
                    "question": r["question"],
                    "response_hash": r["response_hash"],
                    "failure_categories": json.loads(r["failure_categories"]),
                    "comment": r["comment"],
                }
                for r in recent_rows
            ]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "positive": {"count": pos, "pct": pct(pos)},
            "negative": {"count": neg, "pct": pct(neg)},
            "failure_categories": cat_counts,
            "recent_negative": recent,
        }
