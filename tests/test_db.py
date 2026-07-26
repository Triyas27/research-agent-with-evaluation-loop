def test_init_db_creates_runs_table(temp_db):
    import sqlite3

    conn = sqlite3.connect(temp_db.DB_PATH)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    ).fetchall()
    conn.close()
    assert tables == [("runs",)]


def test_log_run_and_get_all_runs_roundtrip(temp_db):
    score_per_iteration = [
        {"iteration": 1, "grounding": 4, "completeness": 3, "coherence": 3, "total": 10, "feedback": "great"}
    ]
    temp_db.log_run(
        run_id="run-1",
        query="compare things",
        status="success",
        iterations=1,
        final_score=10.0,
        score_per_iteration=score_per_iteration,
        num_sources=5,
        total_tokens=1234,
        estimated_cost_usd=0.001,
        latency_seconds=3.2,
        stop_reason="score 10/10 met threshold (7)",
        error_message=None,
    )

    rows = temp_db.get_all_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-1"
    assert row["status"] == "success"
    assert row["final_score"] == 10.0
    # score_per_iteration round-trips through JSON, not stays a raw string
    assert row["score_per_iteration"] == score_per_iteration


def test_get_all_runs_respects_limit(temp_db):
    for i in range(3):
        temp_db.log_run(
            run_id=f"run-{i}",
            query=f"query {i}",
            status="success",
            iterations=1,
            final_score=8.0,
            score_per_iteration=[],
            num_sources=2,
            total_tokens=100,
            estimated_cost_usd=0.0001,
            latency_seconds=1.0,
            stop_reason="ok",
            error_message=None,
        )

    assert len(temp_db.get_all_runs(limit=2)) == 2
    assert len(temp_db.get_all_runs(limit=50)) == 3


def test_log_run_handles_null_final_score_and_error_message(temp_db):
    """The `error` status path logs final_score=None — get_all_runs must not choke on it."""
    temp_db.log_run(
        run_id="run-error",
        query="broken query",
        status="error",
        iterations=0,
        final_score=None,
        score_per_iteration=[],
        num_sources=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        latency_seconds=0.5,
        stop_reason="unhandled exception",
        error_message="boom",
    )
    row = temp_db.get_all_runs()[0]
    assert row["final_score"] is None
    assert row["error_message"] == "boom"
