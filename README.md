# Research Agent with Evaluation Loop — Week 3

Worker (search + draft) + critic (scores 0–10, structured feedback) + retry loop, plus
this week's additions: every run is logged to SQLite, and the agent has guardrails for
timeouts, cost caps, and tool failures instead of just crashing. Dashboard + writeup
come in Week 4.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # Mac/Linux: cp .env.example .env
```

Edit `.env` and fill in:
- `GROQ_API_KEY` — free, from console.groq.com (sign up, create an API key, no billing info needed)
- `TAVILY_API_KEY` — free tier at tavily.com
- `API_KEY` — a shared secret the frontend sends to the backend on `/research`; generate
  one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. This matters
  most once the backend is deployed with a public URL — without it, anyone who finds
  the URL can spend your Groq/Tavily quota.

Everything else in `.env` has a working default:
- `DRAFT_MODEL` / `CRITIC_MODEL` — strong model writes/revises, cheap model scores
- `MAX_ITERATIONS=3`, `SCORE_THRESHOLD=7` — retry loop caps
- `TIMEOUT_SECONDS=60` — wall-clock budget per query; if exceeded, the loop stops after
  its current iteration and returns the best draft so far
- `MAX_COST_USD=0.05` — estimated-cost cap per query (Groq's free tier is actually $0;
  this uses placeholder per-token prices in `main.py` so the cap and the cost numbers
  in the UI/logs are meaningful — adjust `PRICE_PER_1M_TOKENS` if you move to a paid tier)
- `DB_PATH=runs.db` — SQLite file the logging table lives in (created automatically)

## Run

Backend (terminal 1):

```bash
uvicorn main:app --reload --port 8000
```

Frontend (terminal 2):

```bash
streamlit run streamlit_app.py
```

Open the Streamlit URL (usually http://localhost:8501), ask a research question, and
click "Run research". The UI shows score, iterations, latency, estimated cost, why the
loop stopped, and a per-iteration feedback breakdown. The "Recent runs" panel at the
bottom reads straight from the logging DB.

A second page, **Dashboard** (`pages/1_Dashboard.py`), shows up automatically in the
Streamlit sidebar. It aggregates every logged run into: total runs, success rate,
avg iterations, avg cost/query, avg latency, a score-distribution histogram, a
stop-reason breakdown, cost-per-query over time, an iterations histogram, and a
failure log of every run that stopped without meeting `SCORE_THRESHOLD`.

## Guardrails

- **Timeout**: `TIMEOUT_SECONDS` wall-clock budget checked between iterations — if
  exceeded, the loop stops and returns the best-scoring draft seen so far rather than
  continuing to retry.
- **Cost cap**: `MAX_COST_USD` estimated spend checked the same way.
- **Tool failure fallback**: if the Tavily search call fails (after retries), the agent
  returns a clear "search failed" message instead of a 500, skips the critic entirely
  (nothing to grade), and logs the run as `tool_failure`. If a Groq call fails after
  `LLM_MAX_RETRIES` retries, the same pattern applies — return whatever draft exists,
  log it, don't crash the request.
- **Unhandled errors**: any other exception is still logged (status `error`, with the
  message) before the API returns a 500, so nothing silently disappears.
- **Auth + rate limiting**: `POST /research` requires an `X-API-Key` header matching
  `API_KEY` and is rate-limited to 10 requests/minute per client IP (via `slowapi`).
  `GET /runs` is unauthenticated (it just reads the log) but rate-limited to
  30 requests/minute. Without this, a public deployment is an open door to anyone's
  Groq/Tavily bill.

## Logging

Every call to `POST /research` writes one row to the `runs` table in `runs.db`:
`run_id, created_at, query, status, iterations, final_score, score_per_iteration,
num_sources, total_tokens, estimated_cost_usd, latency_seconds, stop_reason,
error_message`. `status` is one of `success`, `timeout`, `cost_cap`, `tool_failure`,
`error`.

Inspect it directly:

```bash
python -c "import db, json; print(json.dumps(db.get_all_runs(), indent=2, default=str))"
```

Or via the API:

```bash
curl http://localhost:8000/runs
```

## Test the API directly

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "Compare pricing models of 5 SaaS project management tools"}'
```

## Automated tests

```bash
pytest tests/
```

The suite mocks the Groq/Tavily clients (no API calls, no cost) and covers:
- `tests/test_db.py` — the SQLite logging roundtrip
- `tests/test_graph_nodes.py` — every guardrail branch of the worker/critic graph in
  isolation: search/draft/critic tool failures, malformed critic JSON, and each stop
  condition (score threshold, timeout, cost cap, max iterations)
- `tests/test_api.py` — the FastAPI endpoints end-to-end: auth, empty-query validation,
  rate limiting, and full success/tool-failure/max-iterations runs through the real
  compiled graph, including a regression test for a bug the suite caught (see below)

**Bug the tests caught:** `route_after_draft`/`route_after_critic` are LangGraph
*conditional-edge* functions, not nodes — mutating `state` inside them doesn't get
committed back to the graph (only a node's return value does). `stop_reason`,
`best_draft`, and `best_score` were being set there, so `stop_reason` came back empty
for every non-tool-failure stop, which meant the `/research` endpoint's
`if "timeout budget" in stop_reason` / `"cost cap" in stop_reason` status checks never
matched — timeout and cost-cap runs were silently logged as `success`. Fixed by moving
all state decisions into `draft_node`/`critic_node` (real nodes); the routing functions
now only read already-committed state.

## How the loop works

```
query
  → worker drafts (search + write, with inline citations)
  → search or draft failed?  → return fallback message, log as tool_failure, stop
  → critic scores 0-10 (grounding /4, completeness /3, coherence /3) + feedback
  → critic failed?  → return current draft, log as tool_failure, stop
  → score >= threshold?  → return report, log as success
  → timeout exceeded, or cost cap hit, or max iterations reached?
      → return best-scoring draft seen, log status accordingly
  → else  → worker revises using critic feedback → back to critic
```

## Next steps (per project plan)

- Week 4 remaining: architecture diagram and tradeoff writeup (accuracy vs. cost
  vs. latency). Dashboard and failure-log section are done (see above).
