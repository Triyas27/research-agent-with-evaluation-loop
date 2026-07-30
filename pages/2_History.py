"""
Full search history — every logged run with its actual result, not just a one-line
summary. Complements the Dashboard (aggregate stats) with per-run detail: the report
that was returned, and the full per-iteration score/draft trail behind it.
"""

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="History", page_icon="📜", layout="wide")
st.title("Search History")
st.caption("Every logged research query and its full result — not just a summary line.")

col_limit, col_refresh = st.columns([4, 1])
with col_limit:
    limit = st.number_input("Runs to load", min_value=10, max_value=5000, value=200, step=10)
with col_refresh:
    st.write("")
    if st.button("Refresh"):
        st.cache_data.clear()


@st.cache_data(ttl=30, show_spinner="Loading history...")
def fetch_runs(limit: int):
    resp = requests.get(f"{BACKEND_URL}/runs", params={"limit": limit}, timeout=30)
    resp.raise_for_status()
    return resp.json()


try:
    runs = fetch_runs(limit)
except requests.exceptions.RequestException as e:
    st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
    st.stop()

if not runs:
    st.info("No runs logged yet. Go run a research query first, then come back.")
    st.stop()

df = pd.DataFrame(runs)
df["created_at"] = pd.to_datetime(df["created_at"])
df = df.sort_values("created_at", ascending=False)

col_search, col_status = st.columns([3, 1])
with col_search:
    search = st.text_input("Filter by query text", placeholder="e.g. pricing")
with col_status:
    statuses = st.multiselect("Status", options=sorted(df["status"].unique()), default=[])

filtered = df
if search:
    filtered = filtered[filtered["query"].str.contains(search, case=False, na=False)]
if statuses:
    filtered = filtered[filtered["status"].isin(statuses)]

st.caption(f"Showing {len(filtered)} of {len(df)} runs.")


def best_iteration(score_history):
    """Same selection rule main.py's critic_node uses: total >= best_score so far,
    ties favor the later iteration."""
    best = None
    best_score = -1
    for s in score_history or []:
        if s.get("total", 0) >= best_score:
            best_score = s.get("total", 0)
            best = s
    return best


for _, row in filtered.iterrows():
    label = f"`{row['created_at']}` — **{row['status']}** — \"{row['query']}\" — {row['final_score']}/10"
    with st.expander(label):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Iterations", row["iterations"])
        c2.metric("Latency", f"{row['latency_seconds']}s")
        c3.metric("Est. cost", f"${row['estimated_cost_usd']:.4f}")
        c4.metric("Tokens", row["total_tokens"])
        st.caption(f"Stopped because: {row['stop_reason']}")

        score_history = row.get("score_per_iteration") or []
        best = best_iteration(score_history)

        if best and best.get("draft"):
            st.markdown("**Report returned (best-scoring iteration):**")
            st.markdown(best["draft"])
        elif pd.notna(row.get("error_message")):
            st.code(row["error_message"])
        else:
            st.caption("No draft recorded for this run.")

        if len(score_history) > 1:
            with st.expander("All iterations"):
                for s in score_history:
                    marker = " ⭐ (best)" if s is best else ""
                    st.markdown(
                        f"**Iteration {s['iteration']}**{marker} — total {s['total']}/10 "
                        f"(grounding {s.get('grounding')}/4, completeness {s.get('completeness')}/3, "
                        f"coherence {s.get('coherence')}/3)"
                    )
                    st.caption(s.get("feedback", ""))
                    if s.get("draft"):
                        st.markdown(s["draft"])
                    st.divider()
