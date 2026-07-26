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
    monkeypatch.setattr(main.tavily_client, "search", lambda **kw: {"results": SOURCES})

    def fake_create(**kw):
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
    def boom(**kw):
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
    monkeypatch.setattr(main.tavily_client, "search", lambda **kw: {"results": SOURCES})

    low_score_json = '{"grounding": 1, "completeness": 0, "coherence": 0, "feedback": "weak"}'

    def fake_create(**kw):
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


def test_runs_endpoint_is_rate_limited(api_client):
    statuses = []
    for _ in range(40):
        resp = api_client.get("/runs")
        statuses.append(resp.status_code)
        if resp.status_code == 429:
            break
    assert 429 in statuses
