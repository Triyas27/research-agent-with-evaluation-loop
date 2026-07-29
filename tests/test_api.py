import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import db
import main
from conftest import make_groq_response

SOURCES = [{"title": "A", "url": "http://a.example", "content": "some content"}]
GOOD_CRITIC_JSON = '{"grounding": 4, "completeness": 3, "coherence": 3, "feedback": "Solid."}'


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "api_test.db"))
    # slowapi's limiter storage is a module-global that outlives any single test, and
    # TestClient always presents as the same client ("testclient") — reset it per test
    # so one test's rate-limit test doesn't 429 the next test's first request.
    main.limiter._storage.reset()
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def mock_happy_path(monkeypatch):
    """Search succeeds, draft succeeds, critic scores above threshold first try."""
    async def fake_search(**kw):
        return {"results": SOURCES}

    monkeypatch.setattr(main.tavily_client, "search", fake_search)

    async def fake_create(**kw):
        if kw["model"] == main.CRITIC_MODEL:
            return make_groq_response(GOOD_CRITIC_JSON)
        return make_groq_response("A well-cited report [1].")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", fake_create)


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_research_requires_api_key(api_client):
    resp = api_client.post("/research", json={"query": "compare things"})
    assert resp.status_code == 401


def test_research_rejects_wrong_api_key(api_client):
    resp = api_client.post(
        "/research", json={"query": "compare things"}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


def test_research_rejects_empty_query(api_client):
    resp = api_client.post(
        "/research", json={"query": "   "}, headers={"X-API-Key": "test-shared-key"}
    )
    assert resp.status_code == 400


def test_research_success_path_logs_and_returns_report(api_client, mock_happy_path):
    resp = api_client.post(
        "/research",
        json={"query": "compare things"},
        headers={"X-API-Key": "test-shared-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_score"] == 10
    assert data["iterations"] == 1
    assert "met threshold" in data["stop_reason"]

    runs = api_client.get("/runs").json()
    assert len(runs) == 1
    assert runs[0]["status"] == "success"


def test_research_tool_failure_returns_200_not_500(api_client, monkeypatch):
    async def boom(**kw):
        raise RuntimeError("tavily is down")

    monkeypatch.setattr(main.tavily_client, "search", boom)

    resp = api_client.post(
        "/research",
        json={"query": "compare things"},
        headers={"X-API-Key": "test-shared-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "search tool failed" in data["report"]

    runs = api_client.get("/runs").json()
    assert runs[0]["status"] == "tool_failure"


def test_research_is_rate_limited(api_client, mock_happy_path):
    statuses = []
    for _ in range(20):
        resp = api_client.post(
            "/research",
            json={"query": "compare things"},
            headers={"X-API-Key": "test-shared-key"},
        )
        statuses.append(resp.status_code)
        if resp.status_code == 429:
            break
    assert 429 in statuses


def test_research_exhausts_max_iterations_when_score_never_clears_threshold(
    api_client, monkeypatch
):
    """Regression test: previously stop_reason was set inside a LangGraph conditional
    edge function, whose state mutations never made it back into the graph's final
    state, so this scenario's stop_reason came back empty and iterations silently
    looked identical to a fresh/never-revised run."""
    async def fake_search(**kw):
        return {"results": SOURCES}

    monkeypatch.setattr(main.tavily_client, "search", fake_search)

    low_score_json = '{"grounding": 1, "completeness": 0, "coherence": 0, "feedback": "weak"}'

    async def fake_create(**kw):
        if kw["model"] == main.CRITIC_MODEL:
            return make_groq_response(low_score_json)
        return make_groq_response("A weakly-cited report [1].")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", fake_create)

    resp = api_client.post(
        "/research",
        json={"query": "compare things"},
        headers={"X-API-Key": "test-shared-key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["iterations"] == main.MAX_ITERATIONS
    assert "max iterations" in data["stop_reason"]
    assert len(data["score_history"]) == main.MAX_ITERATIONS


def _parse_sse_events(response_text):
    events = []
    for chunk in response_text.strip().split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data: "):
            events.append(json.loads(chunk[len("data: "):]))
    return events


def test_research_stream_requires_api_key(api_client):
    resp = api_client.post("/research/stream", json={"query": "compare things"})
    assert resp.status_code == 401


def test_research_stream_rejects_empty_query(api_client):
    resp = api_client.post(
        "/research/stream", json={"query": "  "}, headers={"X-API-Key": "test-shared-key"}
    )
    assert resp.status_code == 400


def test_research_stream_success_path_emits_progress_then_done(api_client, mock_happy_path):
    resp = api_client.post(
        "/research/stream",
        json={"query": "compare things"},
        headers={"X-API-Key": "test-shared-key"},
    )
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)

    progress_events = [e for e in events if e["event"] == "progress"]
    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1
    assert len(progress_events) >= 2  # at least search + draft + critic in some form

    # search -> draft -> critic, in that order, with human-readable messages
    assert "sources" in progress_events[0]["message"]
    assert any("scored" in e["message"] for e in progress_events)

    data = done_events[0]["data"]
    assert data["final_score"] == 10
    assert "met threshold" in data["stop_reason"]

    runs = api_client.get("/runs").json()
    assert runs[0]["status"] == "success"


def test_research_stream_tool_failure_emits_error_progress_and_done(api_client, monkeypatch):
    async def boom(**kw):
        raise RuntimeError("tavily is down")

    monkeypatch.setattr(main.tavily_client, "search", boom)

    resp = api_client.post(
        "/research/stream",
        json={"query": "compare things"},
        headers={"X-API-Key": "test-shared-key"},
    )
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)

    progress_events = [e for e in events if e["event"] == "progress"]
    done_events = [e for e in events if e["event"] == "done"]
    assert "failed" in progress_events[0]["message"]
    assert len(done_events) == 1
    assert "search tool failed" in done_events[0]["data"]["report"]

    runs = api_client.get("/runs").json()
    assert runs[0]["status"] == "tool_failure"


async def test_concurrent_research_requests_run_in_parallel(tmp_path, monkeypatch):
    """Regression guard for the async conversion: /research awaits AsyncGroq/
    AsyncTavilyClient end-to-end via worker_graph.ainvoke(), so concurrent requests
    should overlap on the event loop instead of serializing. If a blocking call ever
    sneaks back into this path, concurrent requests would take roughly
    n_concurrent * (time per request) instead of ~(time per request)."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "concurrency_test.db"))
    db.init_db()
    main.limiter._storage.reset()

    delay = 0.5

    async def slow_search(**kw):
        await asyncio.sleep(delay)
        return {"results": SOURCES}

    async def slow_create(**kw):
        await asyncio.sleep(delay)
        if kw["model"] == main.CRITIC_MODEL:
            return make_groq_response(GOOD_CRITIC_JSON)
        return make_groq_response("A well-cited report [1].")

    monkeypatch.setattr(main.tavily_client, "search", slow_search)
    monkeypatch.setattr(main.groq_client.chat.completions, "create", slow_create)

    n_concurrent = 5
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.perf_counter()
        responses = await asyncio.gather(
            *[
                client.post(
                    "/research",
                    json={"query": f"query {i}"},
                    headers={"X-API-Key": "test-shared-key"},
                    timeout=30,
                )
                for i in range(n_concurrent)
            ]
        )
        elapsed = time.perf_counter() - start

    assert all(r.status_code == 200 for r in responses)
    # Each request does 2 slow calls (search + draft; critic scores 10/10 first try,
    # so no revision). Fully serialized would take n_concurrent * 2 * delay = 5s;
    # truly concurrent takes roughly 2 * delay regardless of n_concurrent.
    assert elapsed < n_concurrent * delay


def test_runs_endpoint_is_rate_limited(api_client):
    statuses = []
    for _ in range(40):
        resp = api_client.get("/runs")
        statuses.append(resp.status_code)
        if resp.status_code == 429:
            break
    assert 429 in statuses
