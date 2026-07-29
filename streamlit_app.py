"""
Minimal Streamlit UI for the Week 3 worker + critic + logging/guardrails agent.

Run the FastAPI backend (main.py) first, then run this app separately.
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("Research Agent with Evaluation Loop")
st.caption(
    "Week 3 milestone: worker drafts, critic scores it, worker revises until the "
    "score clears the bar — with logging and guardrails (timeout, cost cap, "
    "tool-failure fallback)"
)
st.page_link("pages/1_Dashboard.py", label="View run dashboard", icon="📊")

query = st.text_area(
    "Research question",
    placeholder="e.g. Compare pricing models of 5 SaaS project management tools",
    height=100,
)

if st.button("Run research", type="primary"):
    if not query.strip():
        st.warning("Enter a research question first.")
    else:
      with st.spinner("Searching, drafting, and self-critiquing..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/research",
                json={"query": query},
                headers={"X-API-Key": API_KEY},
                timeout=240,
            )
            resp.raise_for_status()
            data = resp.json()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final score", f"{data['final_score']:.1f}/10")
            col2.metric("Iterations", data["iterations"])
            col3.metric("Latency", f"{data['latency_seconds']}s")
            col4.metric("Est. cost", f"${data['estimated_cost_usd']:.4f}")
            st.caption(
                f"Stopped because: {data['stop_reason']}  ·  "
                f"{data['total_tokens']} tokens  ·  run_id {data['run_id'][:8]}"
            )

            with st.expander("Score history (per iteration)", expanded=True):
                for s in data["score_history"]:
                    st.markdown(
                        f"**Iteration {s['iteration']}** — total {s['total']:.1f}/10 "
                        f"(grounding {s['grounding']:.1f}/4, completeness {s['completeness']:.1f}/3, "
                        f"coherence {s['coherence']:.1f}/3)"
                    )
                    st.caption(s["feedback"])
                    if s.get("draft"):
                        with st.expander(f"Draft as of iteration {s['iteration']}"):
                            st.markdown(s["draft"])

            st.divider()
            st.markdown(data["report"])
        except requests.exceptions.RequestException as e:
            st.error(f"Request to backend failed: {e}")

with st.expander("Recent runs (from the logging DB)"):
    st.caption(
        "A quick raw feed of the latest calls. For aggregate stats, charts, and the "
        "failure log, see the Dashboard page linked above."
    )
    if st.button("Refresh run log"):
        try:
            runs = requests.get(f"{BACKEND_URL}/runs", timeout=30).json()
            if not runs:
                st.caption("No runs logged yet.")
            for r in runs:
                st.markdown(
                    f"`{r['created_at']}` — **{r['status']}** — "
                    f"\"{r['query']}\" — score {r['final_score']}/10, "
                    f"{r['iterations']} iter, {r['latency_seconds']}s, "
                    f"${r['estimated_cost_usd']:.4f}"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Could not load run log: {e}")
