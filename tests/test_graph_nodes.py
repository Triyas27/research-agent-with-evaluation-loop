import json
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


def _async_return(value):
    async def _fn(**kw):
        return value

    return _fn


def _async_raise(exc):
    async def _fn(**kw):
        raise exc

    return _fn


# --- search_node -------------------------------------------------------------
# search_node/draft_node/critic_node are async (real Groq/Tavily calls), so these
# tests are async too — see pytest.ini (asyncio_mode = auto).

async def test_search_node_success(monkeypatch):
    monkeypatch.setattr(
        main.tavily_client, "search", _async_return({"results": SOURCES})
    )
    state = await main.search_node(make_state())
    assert state["search_results"] == SOURCES
    assert state["search_failed"] is False


async def test_search_node_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(main.tavily_client, "search", _async_raise(RuntimeError("tavily is down")))
    state = await main.search_node(make_state())
    assert state["search_results"] == []
    assert state["search_failed"] is True
    assert "search tool failed" in state["stop_reason"]


# --- draft_node ----------------------------------------------------------------
# draft_node is where stop_reason / best_draft / best_score get set for the failure
# paths, not route_after_draft — LangGraph conditional-edge functions don't commit
# state mutations back to the graph, only node return values do.

async def test_draft_node_no_sources_after_search_failure(monkeypatch):
    state = make_state(search_failed=True, stop_reason="search tool failed: boom")
    result = await main.draft_node(state)
    assert result["iteration"] == 1
    assert "search tool failed" in result["draft"]
    # search_node already set stop_reason — draft_node must not overwrite it
    assert result["stop_reason"] == "search tool failed: boom"
    assert result["best_draft"] == result["draft"]
    assert result["best_score"] == 0


async def test_draft_node_no_sources_without_search_failure(monkeypatch):
    state = make_state()  # search_results empty, search_failed False
    result = await main.draft_node(state)
    assert "No search results were returned" in result["draft"]
    assert result["stop_reason"] == "no usable sources"
    assert result["best_draft"] == result["draft"]
    assert result["best_score"] == 0


async def test_draft_node_first_iteration_success(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response("Drafted report [1].")),
    )
    state = make_state(search_results=SOURCES)
    result = await main.draft_node(state)
    assert result["iteration"] == 1
    assert result["draft"] == "Drafted report [1]."
    assert result["draft_failed"] is False
    assert result["total_tokens"] == 100
    assert result["estimated_cost_usd"] > 0


async def test_draft_node_prompt_warns_against_prompt_injection(monkeypatch):
    """Sources are live, unsanitized web scrapes - a page containing "ignore
    previous instructions" text becomes part of the model's context. Locks in that
    the mitigation (delimiter + explicit warning) is actually present in the prompt
    sent to the model, for both the first draft and revisions."""
    captured = {}

    async def fake_create(**kw):
        captured["prompt"] = kw["messages"][0]["content"]
        return make_groq_response("Drafted report [1].")

    monkeypatch.setattr(main.groq_client.chat.completions, "create", fake_create)
    state = make_state(search_results=SOURCES)
    await main.draft_node(state)

    prompt = captured["prompt"]
    assert "<untrusted_web_content>" in prompt
    assert "not instructions" in prompt


async def test_draft_node_revision_uses_prior_feedback(monkeypatch):
    captured = {}

    async def fake_create(**kw):
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
    result = await main.draft_node(state)
    assert result["iteration"] == 2
    assert result["draft"] == "Revised report [1]."
    assert "add more citations" in captured["prompt"]


async def test_draft_node_llm_failure_falls_back(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", _async_raise(RuntimeError("groq is down"))
    )
    state = make_state(search_results=SOURCES)
    result = await main.draft_node(state)
    assert result["draft_failed"] is True
    assert "drafting model failed" in result["stop_reason"]
    assert "try again shortly" in result["draft"]
    assert result["best_draft"] == result["draft"]
    assert result["best_score"] == 0


async def test_draft_node_llm_failure_keeps_existing_draft_text(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", _async_raise(RuntimeError("groq is down"))
    )
    state = make_state(
        search_results=SOURCES,
        iteration=1,
        draft="Previous good draft",
        score_history=[{"iteration": 1, "grounding": 2, "completeness": 1, "coherence": 1, "total": 4, "feedback": "meh"}],
        best_score=4,
        best_draft="Previous good draft",
    )
    result = await main.draft_node(state)
    assert result["draft_failed"] is True
    # existing draft text is not clobbered by the fallback message
    assert result["draft"] == "Previous good draft"


async def test_draft_node_does_not_clobber_earlier_best_on_later_failure(monkeypatch):
    """A revision's draft call failing shouldn't wipe out a real score from an
    earlier iteration (e.g. iteration 1 scored 5/10, iteration 2's redraft fails —
    we should still return the 5/10 draft, not reset to 0)."""
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", _async_raise(RuntimeError("groq is down"))
    )
    state = make_state(
        search_results=SOURCES,
        iteration=1,
        draft="iteration 1 draft",
        score_history=[{"iteration": 1, "grounding": 2, "completeness": 2, "coherence": 1, "total": 5, "feedback": "ok"}],
        best_score=5,
        best_draft="iteration 1 draft",
    )
    result = await main.draft_node(state)
    assert result["best_score"] == 5
    assert result["best_draft"] == "iteration 1 draft"


# --- route_after_draft -----------------------------------------------------
# Pure reader now — no state mutation, just picks a route based on flags already
# set by search_node/draft_node. Stays sync (no I/O).

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

async def test_critic_node_scores_valid_json(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response(
            '{"unsupported_claims": [], "missing_parts": [], "contradictions": [], "feedback": "Solid."}'
        )),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    assert len(result["score_history"]) == 1
    score = result["score_history"][0]
    assert score["total"] == 10
    assert result["best_score"] == 10
    assert result["best_draft"] == "a draft"


async def test_critic_prompt_warns_against_backslash_escaped_quotes(monkeypatch):
    """Regression test: a live run against Groq failed with json_validate_failed
    because the critic quoted a phrase from the draft using Python/JS-style \\'
    escaping, which isn't valid JSON (single quotes never need escaping). The fix
    was prompt-level guidance; this locks in that the "bad" example actually renders
    a literal backslash (an earlier version of this fix had Python's own string
    escaping silently swallow it, making the good/bad example identical and useless)."""
    captured = {}

    async def fake_create(**kw):
        captured["prompt"] = kw["messages"][0]["content"]
        return make_groq_response(
            '{"unsupported_claims": [], "missing_parts": [], "contradictions": [], "feedback": "ok"}'
        )

    monkeypatch.setattr(main.groq_client.chat.completions, "create", fake_create)
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    await main.critic_node(state)

    prompt = captured["prompt"]
    assert "never need a backslash" in prompt
    assert "draft\\'s claim" in prompt  # the literal bad example must contain a real backslash


async def test_critic_prompt_warns_against_prompt_injection(monkeypatch):
    captured = {}

    async def fake_create(**kw):
        captured["prompt"] = kw["messages"][0]["content"]
        return make_groq_response(
            '{"unsupported_claims": [], "missing_parts": [], "contradictions": [], "feedback": "ok"}'
        )

    monkeypatch.setattr(main.groq_client.chat.completions, "create", fake_create)
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    await main.critic_node(state)

    prompt = captured["prompt"]
    assert "<untrusted_web_content>" in prompt
    assert "not instructions" in prompt


async def test_critic_node_scores_by_counting_enumerated_issues(monkeypatch):
    """Core property of the counting-based rubric: grounding/completeness/coherence
    come from 4/3/3 minus the length of each issue list, not from a number the
    critic reports itself. A holistic "just give me a 0-10 vibes score" design was
    found live to collapse onto ~the same verdict (3/2/2) for any "substantively
    answered but imperfect" draft regardless of how many distinct problems actually
    existed - verified across a dozen+ live runs on wildly different topics. This
    counting design gives the score somewhere to actually vary."""
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response(json.dumps({
            "unsupported_claims": ["claim A", "claim B"],
            "missing_parts": ["part A"],
            "contradictions": [],
            "feedback": "Two unsupported claims, one part unanswered.",
        }))),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    score = result["score_history"][0]
    assert score["grounding"] == 2  # 4 - 2 unsupported claims
    assert score["completeness"] == 2  # 3 - 1 missing part
    assert score["coherence"] == 3  # 3 - 0 contradictions
    assert score["total"] == 7


async def test_critic_node_score_never_goes_negative(monkeypatch):
    """More issues than the category max shouldn't produce a negative score."""
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response(json.dumps({
            "unsupported_claims": ["a", "b", "c", "d", "e", "f"],  # 6 > max of 4
            "missing_parts": [],
            "contradictions": [],
            "feedback": "Badly unsupported.",
        }))),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    assert result["score_history"][0]["grounding"] == 0


async def test_critic_node_coerces_non_list_issue_fields(monkeypatch):
    """If the critic returns a malformed shape (e.g. a string instead of a list) for
    one of the issue fields, treat it as no issues found rather than crashing."""
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response(json.dumps({
            "unsupported_claims": "not a list",
            "missing_parts": [],
            "contradictions": [],
            "feedback": "ok",
        }))),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    assert result["score_history"][0]["grounding"] == 4


async def test_critic_node_records_draft_text_per_iteration(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response(
            '{"unsupported_claims": [], "missing_parts": [], "contradictions": [], "feedback": "Solid."}'
        )),
    )
    state = make_state(search_results=SOURCES, draft="the draft text for this iteration", iteration=1)
    result = await main.critic_node(state)
    assert result["score_history"][0]["draft"] == "the draft text for this iteration"


async def test_critic_node_invalid_json_scores_conservatively(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions,
        "create",
        _async_return(make_groq_response("this is not json")),
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    score = result["score_history"][0]
    assert score["total"] == 0
    assert "not valid JSON" in score["feedback"]


async def test_critic_node_llm_failure_stops_loop(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", _async_raise(RuntimeError("groq is down"))
    )
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    assert result["critic_failed"] is True
    assert "critic model failed" in result["stop_reason"]
    assert result["best_score"] == 0
    assert result["best_draft"] == "a draft"
    assert "Critic call failed" in result["score_history"][0]["feedback"]


async def test_critic_node_llm_failure_does_not_clobber_earlier_best(monkeypatch):
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", _async_raise(RuntimeError("groq is down"))
    )
    state = make_state(
        search_results=SOURCES,
        draft="revision that broke",
        iteration=2,
        best_score=6,
        best_draft="earlier, better draft",
    )
    result = await main.critic_node(state)
    assert result["best_score"] == 6
    assert result["best_draft"] == "earlier, better draft"


def _critic_returning(grounding, completeness, coherence, monkeypatch):
    """Build a fake critic response that will score exactly (grounding, completeness,
    coherence) once critic_node counts issue-list lengths (grounding = 4 -
    len(unsupported_claims), etc.) - callers pass target scores, not raw JSON."""
    content = json.dumps({
        "unsupported_claims": [f"issue {i}" for i in range(4 - grounding)],
        "missing_parts": [f"missing {i}" for i in range(3 - completeness)],
        "contradictions": [f"contradiction {i}" for i in range(3 - coherence)],
        "feedback": "fb",
    })
    monkeypatch.setattr(
        main.groq_client.chat.completions, "create", _async_return(make_groq_response(content))
    )


async def test_critic_node_sets_stop_reason_when_threshold_met(monkeypatch):
    _critic_returning(4, 3, 3, monkeypatch)  # total 10 >= SCORE_THRESHOLD
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    assert "met threshold" in result["stop_reason"]


async def test_critic_node_sets_stop_reason_on_timeout(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)  # total 1 < threshold
    state = make_state(
        search_results=SOURCES,
        draft="a draft",
        iteration=1,
        start_time=time.time() - (main.TIMEOUT_SECONDS + 5),
    )
    result = await main.critic_node(state)
    assert "timeout budget" in result["stop_reason"]


async def test_critic_node_sets_stop_reason_on_cost_cap(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)
    state = make_state(
        search_results=SOURCES,
        draft="a draft",
        iteration=1,
        estimated_cost_usd=main.MAX_COST_USD + 0.01,
    )
    result = await main.critic_node(state)
    assert "cost cap" in result["stop_reason"]


async def test_critic_node_sets_stop_reason_on_max_iterations(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)
    state = make_state(
        search_results=SOURCES, draft="a draft", iteration=main.MAX_ITERATIONS
    )
    result = await main.critic_node(state)
    assert "max iterations" in result["stop_reason"]


async def test_critic_node_leaves_stop_reason_empty_when_should_revise(monkeypatch):
    _critic_returning(1, 0, 0, monkeypatch)
    state = make_state(search_results=SOURCES, draft="a draft", iteration=1)
    result = await main.critic_node(state)
    assert result["stop_reason"] == ""


# --- route_after_critic ------------------------------------------------------
# Pure reader now — critic_node already decided and recorded the reason. Stays sync.

def test_route_after_critic_ends_on_critic_failure():
    state = make_state(critic_failed=True, stop_reason="")
    assert main.route_after_critic(state) == "end"


def test_route_after_critic_ends_when_stop_reason_set():
    state = make_state(stop_reason="score 10/10 met threshold (7.0)")
    assert main.route_after_critic(state) == "end"


def test_route_after_critic_revises_when_stop_reason_empty():
    state = make_state(stop_reason="")
    assert main.route_after_critic(state) == "revise"
