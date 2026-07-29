"""
Research Agent with Evaluation Loop — Week 3 milestone.

Worker agent (search + draft) + critic agent (scores the draft, structured feedback)
+ retry loop (revise up to MAX_ITERATIONS, or until SCORE_THRESHOLD is met) + this
week's additions:

- Logging: every /research call is written to SQLite (db.py) — run_id, query,
  iterations, score per iteration, tokens, estimated cost, latency, stop reason.
- Guardrails: wall-clock timeout budget, cost cap, and fallbacks when the search
  tool or the LLM calls fail, instead of the whole request 500-ing.

Dashboard + writeup come in Week 4.
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from groq import Groq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from tavily import TavilyClient

import db

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
API_KEY = os.environ.get("API_KEY")

# Tradeoff talking point: stronger model for the final draft, cheaper/faster model
# for the critic loop (it runs multiple times per query).
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "llama-3.3-70b-versatile")
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "llama-3.1-8b-instant")

MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "7"))

# --- Guardrails ---------------------------------------------------------------
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "60"))
MAX_COST_USD = float(os.environ.get("MAX_COST_USD", "0.05"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))

# Groq's free tier bills nothing, but we still track *what it would cost* on a paid
# tier so the cost-cap guardrail and the Week 4 writeup have real numbers to show.
# Adjust these to match current Groq pricing if you move off the free tier.
PRICE_PER_1M_TOKENS = {
    DRAFT_MODEL: {"input": 0.59, "output": 0.79},
    CRITIC_MODEL: {"input": 0.05, "output": 0.08},
}

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
if not API_KEY:
    raise RuntimeError(
        "API_KEY is not set. Add it to your .env file (generate one with "
        "`python -c \"import secrets; print(secrets.token_urlsafe(32))\"`)."
    )

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ---------------------------------------------------------------------------
# Agent state + graph
# ---------------------------------------------------------------------------

class CriticScore(TypedDict):
    iteration: int
    grounding: float
    completeness: float
    coherence: float
    total: float
    feedback: str
    draft: str


class AgentState(TypedDict):
    query: str
    search_results: List[Dict[str, Any]]
    draft: str
    iteration: int
    best_draft: str
    best_score: float
    score_history: List[CriticScore]
    stop_reason: str
    start_time: float
    total_tokens: int
    estimated_cost_usd: float
    search_failed: bool
    draft_failed: bool
    critic_failed: bool


def _sources_block(sources: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {r.get('title', 'Untitled')}\n"
        f"URL: {r.get('url', 'unknown')}\n"
        f"Content: {(r.get('content') or '')[:1500]}"
        for i, r in enumerate(sources)
    )


def _call_groq_with_retry(**kwargs):
    """Call the Groq API, retrying transient failures before giving up."""
    last_err: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - genuinely want to catch+retry any SDK error
            last_err = e
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_err


def _track_usage(state: AgentState, model: str, usage) -> None:
    if usage is None:
        return
    state["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
    prices = PRICE_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    cost = (
        (getattr(usage, "prompt_tokens", 0) or 0) / 1_000_000 * prices["input"]
        + (getattr(usage, "completion_tokens", 0) or 0) / 1_000_000 * prices["output"]
    )
    state["estimated_cost_usd"] += cost


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def search_node(state: AgentState) -> AgentState:
    """Search the web for sources. Guardrail: on tool failure, degrade gracefully
    instead of crashing the whole request."""
    try:
        results = tavily_client.search(
            query=state["query"],
            max_results=8,
            search_depth="advanced",
        )
        state["search_results"] = results.get("results", [])
    except Exception as e:  # noqa: BLE001
        state["search_results"] = []
        state["search_failed"] = True
        state["stop_reason"] = f"search tool failed: {e}"
    return state


def draft_node(state: AgentState) -> AgentState:
    """Draft (iteration 1) or revise (iteration > 1) the report."""
    sources = state["search_results"]
    state["iteration"] += 1

    if not sources:
        if state.get("search_failed"):
            state["draft"] = (
                "The web search tool failed, so I can't produce a grounded report "
                "right now. Please try again shortly."
            )
        else:
            state["draft"] = (
                "No search results were returned for this query, so I can't produce "
                "a grounded report. Try rephrasing the question."
            )
            if not state["stop_reason"]:
                state["stop_reason"] = "no usable sources"
        if state["best_score"] < 0:
            state["best_draft"] = state["draft"]
            state["best_score"] = 0
        return state

    sources_block = _sources_block(sources)

    if state["iteration"] == 1:
        prompt = f"""You are a research assistant. Using ONLY the sources below, write a \
structured, well-organized report answering the user's question.

Rules:
- Every factual claim must have an inline citation like [1], [2], referring to the \
numbered sources below.
- If the sources don't cover something, say so explicitly rather than guessing.
- Structure the report with clear sections/headers appropriate to the question.
- End with a "Sources" section listing each number and its URL.

User question: {state['query']}

Sources:
{sources_block}
"""
    else:
        last_score = state["score_history"][-1]
        prompt = f"""You are revising a research report based on critic feedback. Using \
ONLY the sources below, produce a full REVISED report (not a diff) that fixes the \
issues raised.

User question: {state['query']}

Sources:
{sources_block}

Previous draft (scored {last_score['total']}/10):
{state['draft']}

Critic feedback to address:
{last_score['feedback']}

Rules:
- Every factual claim must have an inline citation like [1], [2].
- Structure the report with clear sections/headers.
- End with a "Sources" section listing each number and its URL.
- Return the complete revised report.
"""

    try:
        response = _call_groq_with_retry(
            model=DRAFT_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:  # noqa: BLE001
        state["draft_failed"] = True
        state["stop_reason"] = f"drafting model failed after retries: {e}"
        if not state["draft"]:
            state["draft"] = (
                "The drafting model failed after retrying, and no earlier draft is "
                "available. Please try again shortly."
            )
        if state["best_score"] < 0:
            state["best_draft"] = state["draft"]
            state["best_score"] = 0
        return state

    _track_usage(state, DRAFT_MODEL, response.usage)
    state["draft"] = response.choices[0].message.content
    return state


def route_after_draft(state: AgentState) -> Literal["critic", "end"]:
    """Guardrail: skip the critic entirely if the draft step already failed hard —
    no point scoring a report that couldn't be produced. All state (stop_reason,
    best_draft/best_score) is set inside search_node/draft_node — this function only
    reads state and picks a route, since LangGraph conditional-edge functions don't
    commit state mutations back to the graph (only node return values do)."""
    if state.get("search_failed") or state.get("draft_failed") or not state["search_results"]:
        return "end"
    return "critic"


def critic_node(state: AgentState) -> AgentState:
    """Score the current draft 0-10 on grounding, completeness, coherence."""
    sources_block = _sources_block(state["search_results"])

    prompt = f"""You are a strict critic grading a research report draft. Score it on:

- grounding (0-4): does every factual claim have an inline citation [n] that matches \
a real source below? Deduct heavily for unsupported claims.
- completeness (0-3): does the draft address every part of the user's question?
- coherence (0-3): is it logically structured, well organized, with no contradictions?

Return ONLY valid JSON, no other text, in exactly this format:
{{"grounding": <int 0-4>, "completeness": <int 0-3>, "coherence": <int 0-3>, \
"feedback": "<specific, actionable feedback citing exact issues to fix>"}}

User question: {state['query']}

Sources:
{sources_block}

Draft to grade:
{state['draft']}
"""

    try:
        response = _call_groq_with_retry(
            model=CRITIC_MODEL,
            max_tokens=600,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        _track_usage(state, CRITIC_MODEL, response.usage)
        raw = response.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        # Guardrail: critic tool failure. Keep the current draft as the result
        # rather than crashing, and stop the loop rather than retrying blindly.
        state["critic_failed"] = True
        state["stop_reason"] = f"critic model failed after retries: {e}"
        if state["best_score"] < 0:
            state["best_draft"] = state["draft"]
            state["best_score"] = 0
        state["score_history"].append(
            {
                "iteration": state["iteration"],
                "grounding": 0,
                "completeness": 0,
                "coherence": 0,
                "total": 0,
                "feedback": f"Critic call failed: {e}",
                "draft": state["draft"],
            }
        )
        return state

    try:
        parsed = json.loads(raw)
        # float(), not int(): a critic returning e.g. 3.5 shouldn't get silently
        # floored to 3 — that's a real precision loss in the exact rubric this
        # project is built around.
        grounding = float(parsed.get("grounding", 0))
        completeness = float(parsed.get("completeness", 0))
        coherence = float(parsed.get("coherence", 0))
        feedback = str(parsed.get("feedback", "")).strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        # Guardrail: if the critic doesn't return parseable JSON, score conservatively
        # low so the loop retries (or exhausts iterations) instead of silently trusting
        # an unscored draft.
        grounding, completeness, coherence = 0, 0, 0
        feedback = f"Critic response was not valid JSON, could not be scored: {raw[:300]}"

    total = round(grounding + completeness + coherence, 2)

    score: CriticScore = {
        "iteration": state["iteration"],
        "grounding": grounding,
        "completeness": completeness,
        "coherence": coherence,
        "total": total,
        "feedback": feedback,
        "draft": state["draft"],
    }
    state["score_history"].append(score)

    if total >= state["best_score"]:
        state["best_score"] = total
        state["best_draft"] = state["draft"]

    # Guardrail decisions (threshold / timeout / cost cap / max iterations) are made
    # here, inside the node, rather than in route_after_critic: LangGraph conditional-
    # edge functions don't commit state mutations back to the graph, only node return
    # values do. state["stop_reason"] doubles as the "should we stop?" signal — left
    # empty means keep revising.
    elapsed = time.time() - state["start_time"]

    if total >= SCORE_THRESHOLD:
        state["stop_reason"] = f"score {total}/10 met threshold ({SCORE_THRESHOLD})"
    elif elapsed >= TIMEOUT_SECONDS:
        state["stop_reason"] = (
            f"timeout budget ({TIMEOUT_SECONDS}s) exceeded after {elapsed:.1f}s; "
            f"returning best draft (score {state['best_score']}/10)"
        )
    elif state["estimated_cost_usd"] >= MAX_COST_USD:
        state["stop_reason"] = (
            f"cost cap (${MAX_COST_USD}) reached (${state['estimated_cost_usd']:.4f} spent); "
            f"returning best draft (score {state['best_score']}/10)"
        )
    elif state["iteration"] >= MAX_ITERATIONS:
        state["stop_reason"] = (
            f"hit max iterations ({MAX_ITERATIONS}); "
            f"returning best draft (score {state['best_score']}/10)"
        )

    return state


def route_after_critic(state: AgentState) -> Literal["revise", "end"]:
    """Pure reader: critic_node already decided (and recorded in stop_reason) whether
    a guardrail condition was hit. See critic_node for why the decision lives there."""
    if state.get("critic_failed"):
        return "end"
    if state["stop_reason"]:
        return "end"
    return "revise"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("search", search_node)
    graph.add_node("draft", draft_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "draft")
    graph.add_conditional_edges(
        "draft", route_after_draft, {"critic": "critic", "end": END}
    )
    graph.add_conditional_edges(
        "critic", route_after_critic, {"revise": "draft", "end": END}
    )
    return graph.compile()


worker_graph = build_graph()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Research Agent with Evaluation Loop (Week 3: logging + guardrails)")

# Guardrail: /research is the endpoint that actually spends Groq/Tavily quota, so it
# requires a shared API key (checked below) and is rate-limited per client IP. /runs
# is read-only and cheap (SQLite select) but still rate-limited against abuse.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.on_event("startup")
def _on_startup():
    db.init_db()


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    run_id: str
    query: str
    report: str
    num_sources: int
    latency_seconds: float
    iterations: int
    final_score: float
    score_history: List[CriticScore]
    stop_reason: str
    total_tokens: int
    estimated_cost_usd: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs")
@limiter.limit("30/minute")
def runs(request: Request, limit: int = 50):
    """Recent logged runs — powers the Week 4 dashboard."""
    return db.get_all_runs(limit=limit)


def _build_initial_state(query: str, start: float) -> AgentState:
    return {
        "query": query,
        "search_results": [],
        "draft": "",
        "iteration": 0,
        "best_draft": "",
        "best_score": -1.0,
        "score_history": [],
        "stop_reason": "",
        "start_time": start,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "search_failed": False,
        "draft_failed": False,
        "critic_failed": False,
    }


def _classify_status(result: AgentState) -> str:
    if "timeout budget" in result["stop_reason"]:
        return "timeout"
    if "cost cap" in result["stop_reason"]:
        return "cost_cap"
    if result.get("search_failed") or result.get("draft_failed") or result.get("critic_failed"):
        return "tool_failure"
    return "success"


def _log_error(run_id: str, query: str, latency: float, error: Exception) -> None:
    db.log_run(
        run_id=run_id,
        query=query,
        status="error",
        iterations=0,
        final_score=None,
        score_per_iteration=[],
        num_sources=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        latency_seconds=round(latency, 2),
        stop_reason="unhandled exception",
        error_message=str(error)[:2000],
    )


def _finalize(run_id: str, query: str, result: AgentState, latency: float) -> ResearchResponse:
    """Classify the finished run, log it, and build the API response. Shared by
    /research and /research/stream so the two endpoints can't drift."""
    status = _classify_status(result)

    db.log_run(
        run_id=run_id,
        query=query,
        status=status,
        iterations=result["iteration"],
        final_score=result["best_score"],
        score_per_iteration=result["score_history"],
        num_sources=len(result["search_results"]),
        total_tokens=result["total_tokens"],
        estimated_cost_usd=round(result["estimated_cost_usd"], 6),
        latency_seconds=round(latency, 2),
        stop_reason=result["stop_reason"],
        error_message=None,
    )

    return ResearchResponse(
        run_id=run_id,
        query=query,
        report=result["best_draft"] or result["draft"],
        num_sources=len(result["search_results"]),
        latency_seconds=round(latency, 2),
        iterations=result["iteration"],
        final_score=result["best_score"],
        score_history=result["score_history"],
        stop_reason=result["stop_reason"],
        total_tokens=result["total_tokens"],
        estimated_cost_usd=round(result["estimated_cost_usd"], 6),
    )


def _progress_message(node: str, state: AgentState) -> str:
    if node == "search":
        if state.get("search_failed"):
            return "Web search failed"
        return f"Searched the web, found {len(state['search_results'])} sources"
    if node == "draft":
        if state.get("draft_failed"):
            return f"Drafting failed on iteration {state['iteration']}"
        verb = "Drafted" if state["iteration"] == 1 else "Revised"
        return f"{verb} report (iteration {state['iteration']})"
    if node == "critic":
        if state.get("critic_failed"):
            return "Critic call failed"
        last = state["score_history"][-1]
        return f"Critic scored iteration {state['iteration']}: {last['total']}/10"
    return node


@app.post("/research", response_model=ResearchResponse)
@limiter.limit("10/minute")
def research(request: Request, req: ResearchRequest, _: None = Depends(require_api_key)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    run_id = str(uuid.uuid4())
    start = time.time()
    initial_state = _build_initial_state(req.query, start)

    try:
        result = worker_graph.invoke(initial_state)
    except Exception as e:  # noqa: BLE001 - guardrail: log + fail cleanly, don't 500 silently
        _log_error(run_id, req.query, time.time() - start, e)
        raise HTTPException(status_code=500, detail=f"Research run failed: {e}") from e

    return _finalize(run_id, req.query, result, time.time() - start)


@app.post("/research/stream")
@limiter.limit("10/minute")
def research_stream(request: Request, req: ResearchRequest, _: None = Depends(require_api_key)):
    """Same worker/critic loop as /research, but streamed as Server-Sent Events so
    the UI can show live progress instead of a single silent 60-second wait. Emits
    one 'progress' event per graph node, then a final 'done' event carrying the same
    payload /research returns (or an 'error' event on an unhandled exception)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    def event_stream():
        run_id = str(uuid.uuid4())
        start = time.time()
        initial_state = _build_initial_state(req.query, start)

        final_state: Optional[AgentState] = None
        try:
            for update in worker_graph.stream(initial_state):
                node, state = next(iter(update.items()))
                final_state = state
                yield f"data: {json.dumps({'event': 'progress', 'message': _progress_message(node, state)})}\n\n"
        except Exception as e:  # noqa: BLE001 - same guardrail as /research, just over SSE
            _log_error(run_id, req.query, time.time() - start, e)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            return

        response = _finalize(run_id, req.query, final_state, time.time() - start)
        yield f"data: {json.dumps({'event': 'done', 'data': json.loads(response.model_dump_json())})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
