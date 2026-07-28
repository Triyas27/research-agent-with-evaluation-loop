import time

import pytest

import main
from conftest import make_groq_response


def make_state(**overrides):
    state = {
        "query": "compare things",
        "search_results": [],
        "draft": "",
        "iteration": 0,
        "best_draft": "",
        "best_score": -1.0,
        "score_history": [],
        "stop_reason": "",
        "start_time": time.time(),
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "search_failed": False,
        "draft_failed": False,
        "critic_failed": False,
    }
    state.update(overrides)
    return state


SOURCES = [{"title": "A", "url": "http://a.example", "content": "some content"}]


# --- search_node -------------------------------------------------------------

def test_search_node_success(monkeypatch):
    monkeypatch.setattr(
        main.tavily_client, "search", lambda **kw: {"results": SOURCES}
    )
    state = main.search_node(make_state())
    assert state["search_results"] == SOURCES
    assert state["search_failed"] is False


def test_search_node_failure_degrades_gracefully(monkeypatch):
    def boom(**kw):
        raise RuntimeError("tavily is down")

    monkeypatch.setattr(main.tavily_client, "search", boom)
    state = main.search_node(make_state())
    assert state["search_results"] == []
    assert state["search_failed"] is True
    assert "search tool failed" in state["stop_reason"]


# --- draft_node ----------------------------------------------------------------
# draft_node is where stop_reason / best_draft / best_score get set for the failure
# paths, not route_after_draft — LangGraph conditional-edge functions don't commit
# state mutations back to the graph, only node return values do.

def test_draft_node_no_sources_after_search_failure(monkeypatch):
    state = make_state(search_failed=True, stop_reason="search tool failed: boom")
    result = main.draft_node(state)
    assert result["iteration"] == 1
    assert "search tool failed" in result["draft"]
    # search_node already set stop_reason — draft_node must not overwrite it
    assert result["stop_reason"] == "search tool failed: boom"
    assert result["best_draft"] == result["draft"]
    assert result["best_score"] == 0


def test_draft_node_no_sources_without_search_failure(monkeypatch):
    state = make_state()  # search_results empty, search_failed False
    result = main.draft_node(state)
    assert "No search results were returned" in result["draft"]
    assert result["stop_reason"] == "no usable sources"
    assert result["best_draft"] == result["draft"]
    assert result["best_score"] == 0


def test_draft_node_first_iteration_success(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        lambda **kw: make_groq_response("Drafted report [1]."),
    )
    state = make_state(search_results=SOURCES)
    result = main.draft_node(state)
    assert result["iteration"] == 1
    assert result["draft"] == "Drafted report [1]."
    assert result["draft_failed"] is False
    assert result["total_tokens"] == 100
    assert result["estimated_cost_usd"] > 0


def test_draft_node_revision_uses_prior_feedback(monkeypatch):
    captured = {}

    def fake_create(**kw):
        captured["prompt"] = kw["messages"][0]["content"]
        return make_groq_response("Revised report [1].")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", fake_create)
    state = make_state(
        search_results=SOURCES,
        iteration=1,
        draft="Old draft",
        score_history=[
            {"iteration": 1, "grounding": 2, "completeness": 1, "coherence": 1, "total": 4, "feedback": "add more citations"}
        ],
    )
    result = main.draft_node(state)
    assert result["iteration"] == 2
    assert result["draft"] == "Revised report [1]."
    assert "add more citations" in captured["prompt"]


def test_draft_node_llm_failure_falls_back(monkeypatch):
    def boom(**kw):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", boom)
    state = make_state(search_results=SOURCES)
    result = main.draft_node(state)
    assert result["draft_failed"] is True
    assert "drafting model failed" in result["stop_reason"]
    assert "try again shortly" in result["draft"]
    assert result["best_draft"] == result["draft"]
    assert result["best_score"] == 0


def test_draft_node_llm_failure_keeps_existing_draft_text(monkeypatch):
    def boom(**kw):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", boom)
    state = make_state(
        search_results=SOURCES,
        iteration=1,
        draft="Previous good draft",
        score_history=[{"iteration": 1, "grounding": 2, "completeness": 1, "coherence": 1, "total": 4, "feedback": "meh"}],
        best_score=4,
        best_draft="Previous good draft",
    )
    result = main.draft_node(state)
    assert result["draft_failed"] is True
    # existing draft text is not clobbered by the fallback message
    assert result["draft"] == "Previous good draft"


def test_draft_node_does_not_clobber_earlier_best_on_later_failure(monkeypatch):
    """A revision's draft call failing shouldn't wipe out a real score from an
    earlier iteration (e.g. iteration 1 scored 5/10, iteration 2's redraft fails —
    we should still return the 5/10 draft, not reset to 0)."""
    def boom(**kw):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", boom)
    state = make_state(
        search_results=SOURCES,
        iteration=1,
        draft="iteration 1 draft",
        score_history=[{"iteration": 1, "grounding": 2, "completeness": 2, "coherence": 1, "total": 5, "feedback": "ok"}],
        best_score=5,
        best_draft="iteration 1 draft",
    )
    result = main.draft_node(state)
    assert result["best_score"] == 5
    assert result["best_draft"] == "iteration 1 draft"


# --- route_after_draft -----------------------------------------------------
# Pure reader now — no state mutation, just picks a route based on flags already
# set by search_node/draft_node.

def test_route_after_draft_goes_to_critic_when_healthy():
    state = make_state(search_results=SOURCES, draft="a draft")
    assert main.route_after_draft(state) == "critic"


@pytest.mark.parametrize("flag", ["search_failed", "draft_failed"])
def test_route_after_draft_ends_on_failure_flags(flag):
    state = make_state(search_results=SOURCES, draft="fallback message", **{flag: True})
    assert main.route_after_draft(state) == "end"


def test_route_after_draft_ends_when_no_sources():
    state = make_state(search_results=[], draft="no sources fallback")
    assert main.route_after_draft(state) == "end"


# --- critic_node -------------------------------------------------------------
# critic_node now also owns the stop/revise guardrail decision (threshold, timeout,
# cost cap, max iterations) — it records the reason in stop_reason, which
# route_after_critic then just reads.

def test_critic_node_scores_valid_json(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        lambda **kw: make_groq_response(
            '{"grounding": 4, "completeness": 3, "coherence": 3, "feedback": "Solid."}'
        ),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = main.critic_node(state)
    assert len(result["score_history"]) == 1
    score = result["score_history"][0]
    assert score["total"] == 10
    assert result["best_score"] == 10
    assert result["best_draft"] == "a draft"


def test_critic_node_preserves_fractional_scores(monkeypatch):
    """A critic returning 3.5 must not be silently floored to 3 — that's the exact
    rubric this project's whole pitch is built around."""
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        lambda **kw: make_groq_response(
            '{"grounding": 3.5, "completeness": 2.5, "coherence": 2, "feedback": "Mostly solid."}'
        ),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = main.critic_node(state)
    score = result["score_history"][0]
    assert score["grounding"] == 3.5
    assert score["completeness"] == 2.5
    assert score["total"] == 8.0


def test_critic_node_records_draft_text_per_iteration(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        lambda **kw: make_groq_response(
            '{"grounding": 4, "completeness": 3, "coherence": 3, "feedback": "Solid."}'
        ),
    )
    state = make_state(search_results=SOURCES, draft="the draft text for this iteration", iteration=1)
    result = main.critic_node(state)
    assert result["score_history"][0]["draft"] == "the draft text for this iteration"


def test_critic_node_invalid_json_scores_conservatively(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        lambda **kw: make_groq_response("this is not json"),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = main.critic_node(state)
    score = result["score_history"][0]
    assert score["total"] == 0
    assert "not valid JSON" in score["feedback"]


def test_critic_node_llm_failure_stops_loop(monkeypatch):
    def boom(**kw):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", boom)
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = main.critic_node(state)
    assert result["critic_failed"] is True
    assert "critic model failed" in result["stop_reason"]
    assert result["best_score"] == 0
    assert result["best_draft"] == "a draft"
    assert "Critic call failed" in result["score_history"][0]["feedback"]


def test_critic_node_llm_failure_does_not_clobber_earlier_best(monkeypatch):
    def boom(**kw):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", boom)
    state = make_state(
        search_results=SOURCES,
        draft="revision that broke",
        iteration=2,
        best_score=6,
        best_draft="earlier, better draft",
    )
    result = main.critic_node(state)
    assert result["best_score"] == 6
    assert result["best_draft"] == "earlier, better draft"


def _critic_returning(total_grounding, total_completeness, total_coherence, monkeypatch):
    content = (
        f'{{"grounding": {total_grounding}, "completeness": {total_completeness}, '
        f'"coherence": {total_coherence}, "feedback": "fb"}}'
    )
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", lambda **kw: make_groq_response(content)
    )


def test_critic_node_sets_stop_reason_when_threshold_met(monkeypatch):
    _critic_returning(4, 3, 3, monkeypatch)  # total 10 >= SCORE_THRESHOLD
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = main.critic_node(state)
    assert "met threshold" in result["stop_reason"]


def test_critic_node_sets_stop_reason_on_timeout(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)  # total 1 < threshold
    state = make_state(
        search_results=SOURCES,
        draft="a draft",
        iteration=1,
        start_time=time.time() - (main.TIMEOUT_SECONDS + 5),
    )
    result = main.critic_node(state)
    assert "timeout budget" in result["stop_reason"]


def test_critic_node_sets_stop_reason_on_cost_cap(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)
    state = make_state(
        search_results=SOURCES,
        draft="a draft",
        iteration=1,
        estimated_cost_usd=main.MAX_COST_USD + 0.01,
    )
    result = main.critic_node(state)
    assert "cost cap" in result["stop_reason"]


def test_critic_node_sets_stop_reason_on_max_iterations(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)
    state = make_state(
        search_results=SOURCES, draft="a draft", iteration=main.MAX_ITERATIONS
    )
    result = main.critic_node(state)
    assert "max iterations" in result["stop_reason"]


def test_critic_node_leaves_stop_reason_empty_when_should_revise(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = main.critic_node(state)
    assert result["stop_reason"] == ""


# --- route_after_critic ------------------------------------------------------
# Pure reader now — critic_node already decided and recorded the reason.

def test_route_after_critic_ends_on_critic_failure():
    state = make_state(critic_failed=True, stop_reason="")
    assert main.route_after_critic(state) == "end"


def test_route_after_critic_ends_when_stop_reason_set():
    state = make_state(stop_reason="score 10/10 met threshold (7.0)")
    assert main.route_after_critic(state) == "end"


def test_route_after_critic_revises_when_stop_reason_empty():
    state = make_state(stop_reason="")
    assert main.route_after_critic(state) == "revise"
