"""
SQLite logging for research agent runs — the observability layer called out in the
project plan as "the differentiator, don't skip it."

One row per /research call: what was asked, how many worker<->critic iterations it
took, the score trajectory, tokens/cost spent, latency, and how/why it stopped
(score threshold, timeout, cost cap, tool failure, or error).
"""

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = os.environ.get("DB_PATH", "runs.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    status TEXT NOT NULL,              -- success | timeout | cost_cap | tool_failure | error
    iterations INTEGER,
    final_score REAL,
    score_per_iteration TEXT,          -- JSON list of {iteration, grounding, completeness, coherence, total, feedback}
    num_sources INTEGER,
    total_tokens INTEGER,
    estimated_cost_usd REAL,
    latency_seconds REAL,
    stop_reason TEXT,
    error_message TEXT
);
"""


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def log_run(
    *,
    run_id: str,
    query: str,
    status: str,
    iterations: int,
    final_score,
    score_per_iteration,
    num_sources: int,
    total_tokens: int,
    estimated_cost_usd: float,
    latency_seconds: float,
    stop_reason: str,
    error_message: str = None,
) -> None:
    if not isinstance(score_per_iteration, str):
        score_per_iteration = json.dumps(score_per_iteration)

    row = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "status": status,
        "iterations": iterations,
        "final_score": final_score,
        "score_per_iteration": score_per_iteration,
        "num_sources": num_sources,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "latency_seconds": latency_seconds,
        "stop_reason": stop_reason,
        "error_message": error_message,
    }
    columns = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            f"INSERT INTO runs ({columns}) VALUES ({placeholders})",
            list(row.values()),
        )
        conn.commit()


def get_all_runs(limit: int = 50):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["score_per_iteration"] = json.loads(d["score_per_iteration"])
            except (TypeError, json.JSONDecodeError):
                pass
            results.append(d)
        return results
