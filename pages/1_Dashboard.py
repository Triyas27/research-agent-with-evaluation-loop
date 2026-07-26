"""
Week 4 dashboard — reads the run log from the FastAPI backend (`/runs`, backed by
db.py) and turns it into the "what I learned" artifact called for in the project
plan: avg iterations, cost/query, score distribution, and a failure-log section.

Run the FastAPI backend first, then `streamlit run streamlit_app.py` — this page
shows up automatically in the sidebar nav.
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "7"))

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Run Dashboard")
st.caption(
    "Aggregate view over every /research call logged to the run DB — "
    "iterations, cost, score distribution, and where the agent struggled."
)

limit = st.number_input("Runs to load", min_value=10, max_value=5000, value=500, step=10)

try:
    runs = requests.get(f"{BACKEND_URL}/runs", params={"limit": limit}, timeout=30).json()
except requests.exceptions.RequestException as e:
    st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
    st.stop()

if not runs:
    st.info("No runs logged yet. Go run some research queries first, then come back.")
    st.stop()

df = pd.DataFrame(runs)
df["created_at"] = pd.to_datetime(df["created_at"])
df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

# --- Summary metrics -----------------------------------------------------------

total_runs = len(df)
success_rate = (df["status"] == "success").mean() * 100
avg_iterations = df["iterations"].mean()
avg_cost = df["estimated_cost_usd"].mean()
avg_latency = df["latency_seconds"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total runs", total_runs)
c2.metric("Success rate", f"{success_rate:.0f}%")
c3.metric("Avg iterations", f"{avg_iterations:.2f}")
c4.metric("Avg cost/query", f"${avg_cost:.4f}")
c5.metric("Avg latency", f"{avg_latency:.1f}s")

st.divider()

# --- Charts ----------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score distribution")
    fig = px.histogram(
        df.dropna(subset=["final_score"]),
        x="final_score",
        nbins=11,
        range_x=[0, 10],
    )
    fig.add_vline(x=SCORE_THRESHOLD, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title="Final score (/10)", yaxis_title="Runs")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Stop reason breakdown")
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.pie(status_counts, names="status", values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Cost per query over time")
    fig = px.line(
        df.sort_values("created_at"), x="created_at", y="estimated_cost_usd", markers=True
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Estimated cost (USD)")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Iterations distribution")
    fig = px.histogram(df, x="iterations", nbins=int(df["iterations"].max() or 1))
    fig.update_layout(xaxis_title="Iterations", yaxis_title="Runs")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Failure log -------------------------------------------------------------
# Plan calls for: "cases where the agent never scored well even after 3 tries,
# and analysis of why" — a non-`success` status means the loop stopped without
# clearing SCORE_THRESHOLD (timeout/cost_cap/tool_failure/error).

st.subheader("Failure log")
st.caption(
    f"Runs that stopped without meeting the score threshold ({SCORE_THRESHOLD}/10)."
)

failures = df[df["status"] != "success"].sort_values("created_at", ascending=False)

if failures.empty:
    st.success("Every logged run met the score threshold.")
else:
    st.dataframe(
        failures[
            [
                "created_at",
                "query",
                "status",
                "iterations",
                "final_score",
                "stop_reason",
                "error_message",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Failure breakdown by status"):
        for status, group in failures.groupby("status"):
            st.markdown(f"**{status}** — {len(group)} run(s)")
            for _, row in group.iterrows():
                st.caption(f"\"{row['query']}\" — {row['stop_reason']}")
