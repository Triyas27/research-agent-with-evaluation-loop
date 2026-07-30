"""
Minimal Streamlit UI for the Week 3 worker + critic + logging/guardrails agent.

Run the FastAPI backend (main.py) first, then run this app separately.
"""

import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")

st.set_page_config(page_title="Research Agent", page_icon="🔎")
st.title("Research Agent with Evaluation Loop")
st.caption(
    "Week 3 milestone: worker drafts, critic scores it, worker revises until the "
    "score clears the bar — with logging and guardrails (timeout, cost cap, "
    "tool-failure fallback)"
)
st.caption("See **Dashboard** and **History** in the sidebar for aggregate stats and past runs.")

query = st.text_area(
    "Research question",
    placeholder="e.g. Compare pricing models of 5 SaaS project management tools",
    height=100,
)

if st.button("Run research", type="primary"):
    if not query.strip():
        st.warning("Enter a research question first.")
    else:
        try:
            data = None
            with st.status("Starting…", expanded=True) as status:
                resp = requests.post(
                    f"{BACKEND_URL}/research/stream",
                    json={"query": query},
                    headers={"X-API-Key": API_KEY},
                    stream=True,
                    timeout=240,
                )
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = json.loads(line[len("data: "):])
                    event = payload.get("event")
                    if event == "progress":
                        status.write(payload["message"])
                    elif event == "error":
                        status.update(label="Failed", state="error")
                        st.error(payload["message"])
                    elif event == "done":
                        data = payload["data"]
                        status.update(label="Done", state="complete")

            if data is not None:
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

st.page_link("pages/2_History.py", label="View full search history and past results", icon="📜")
