from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_PATH = PROJECT_ROOT / "results" / "tables" / "pilot100_direct_vs_rag_comparison.csv"
COMPARISON_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "pilot100_direct_vs_rag_comparison.summary.json"
BM25_SUMMARY_PATH = PROJECT_ROOT / "results" / "runs" / "bm25__triviaqa_pilot_v1__top5.summary.json"
GROUNDEDNESS_PATH = PROJECT_ROOT / "results" / "tables" / "pilot100_rag_groundedness_analysis.csv"
GROUNDEDNESS_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "pilot100_rag_groundedness_analysis.summary.json"
MANUAL_REVIEW_PATH = PROJECT_ROOT / "results" / "manual_reviews" / "pilot100_manual_review_final.csv"
MANUAL_REVIEW_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "pilot100_manual_review_final.summary.json"
RAG_RUN_PATH = PROJECT_ROOT / "results" / "runs" / "rag__openai__gpt-5-4-mini__bm25_top5__pilot100.csv"


st.set_page_config(page_title="LLM FactCheck", layout="wide")


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def count_chart(counts: dict, title: str, x_label: str, y_label: str):
    if not counts:
        return None
    chart_df = pd.DataFrame(
        [{"label": label, "count": count} for label, count in counts.items()]
    ).sort_values("count", ascending=False)
    return px.bar(
        chart_df,
        x="count",
        y="label",
        orientation="h",
        title=title,
        labels={"count": x_label, "label": y_label},
        text="count",
    )


def parse_evidence_passages(value: str) -> list[str]:
    if not value:
        return []
    try:
        passages = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if not isinstance(passages, list):
        return [str(passages)]
    return [str(passage) for passage in passages]


comparison_df = load_csv(COMPARISON_PATH)
groundedness_df = load_csv(GROUNDEDNESS_PATH)
manual_review_df = load_csv(MANUAL_REVIEW_PATH)
rag_run_df = load_csv(RAG_RUN_PATH)

comparison_summary = load_json(COMPARISON_SUMMARY_PATH)
bm25_summary = load_json(BM25_SUMMARY_PATH)
groundedness_summary = load_json(GROUNDEDNESS_SUMMARY_PATH)
manual_review_summary = load_json(MANUAL_REVIEW_SUMMARY_PATH)

st.title("LLM FactCheck")
st.caption("Reliability evaluation dashboard for the 100-question TriviaQA pilot.")

required_files = {
    "comparison": COMPARISON_PATH,
    "groundedness": GROUNDEDNESS_PATH,
    "manual review": MANUAL_REVIEW_PATH,
}
missing_files = [name for name, path in required_files.items() if not path.exists()]
if missing_files:
    st.error(f"Missing required result files: {', '.join(missing_files)}")
    st.stop()

main_metrics = comparison_summary or {}
groundedness_metrics = groundedness_summary.get("aggregate_metrics", {})

metric_cols = st.columns(6)
metric_cols[0].metric("Direct EM", percent(main_metrics.get("direct_normalized_exact_match")))
metric_cols[1].metric("RAG EM", percent(main_metrics.get("rag_normalized_exact_match")))
metric_cols[2].metric("Direct F1", f"{main_metrics.get('direct_mean_token_f1', 0):.3f}")
metric_cols[3].metric("RAG F1", f"{main_metrics.get('rag_mean_token_f1', 0):.3f}")
metric_cols[4].metric("BM25 Support@5", percent(bm25_summary.get("gold_support_rate_top_k")))
metric_cols[5].metric("Unsupported Proxy", percent(groundedness_metrics.get("unsupported_answer_proxy_rate_all_rows")))

overview_tab, inspector_tab, manual_tab, groundedness_tab = st.tabs(
    ["Overview", "Question Inspector", "Manual Review", "Groundedness"]
)

with overview_tab:
    st.subheader("Pilot Summary")
    st.write(
        "RAG modestly improves aggregate accuracy over direct prompting, but the row-level "
        "comparison shows regressions, refusals, and answer-granularity issues."
    )

    outcome_chart = count_chart(
        main_metrics.get("outcome_counts", {}),
        "Direct vs RAG Outcomes",
        "Rows",
        "Outcome",
    )
    if outcome_chart is not None:
        st.plotly_chart(outcome_chart, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "System": "Direct LLM",
                    "Normalized EM": main_metrics.get("direct_normalized_exact_match"),
                    "Mean Token F1": main_metrics.get("direct_mean_token_f1"),
                    "Support@5": "",
                },
                {
                    "System": "BM25 Retrieval",
                    "Normalized EM": "",
                    "Mean Token F1": "",
                    "Support@5": bm25_summary.get("gold_support_rate_top_k"),
                },
                {
                    "System": "BM25 + RAG",
                    "Normalized EM": main_metrics.get("rag_normalized_exact_match"),
                    "Mean Token F1": main_metrics.get("rag_mean_token_f1"),
                    "Support@5": "",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with inspector_tab:
    st.subheader("Question Inspector")

    outcomes = sorted(comparison_df["comparison_outcome"].dropna().unique())
    selected_outcomes = st.multiselect(
        "Filter by comparison outcome",
        outcomes,
        default=outcomes,
    )
    filtered_df = comparison_df[comparison_df["comparison_outcome"].isin(selected_outcomes)].copy()

    question_options = (
        filtered_df["question_id"].astype(str)
        + " | "
        + filtered_df["question"].astype(str).str.slice(0, 90)
    ).tolist()
    if not question_options:
        st.warning("No rows match the current filters.")
    else:
        selected_option = st.selectbox("Select a question", question_options)
        selected_question_id = selected_option.split(" | ", 1)[0]
        selected_row = filtered_df[filtered_df["question_id"].astype(str) == selected_question_id].iloc[0]

        st.markdown(f"**Question:** {selected_row['question']}")
        st.markdown(f"**Gold answer:** `{selected_row['gold_primary']}`")

        answer_cols = st.columns(2)
        answer_cols[0].markdown("#### Direct Answer")
        answer_cols[0].write(selected_row["direct_answer"])
        answer_cols[0].write(f"Label: `{selected_row['direct_correctness_label']}`")
        answer_cols[0].write(f"Token F1: `{selected_row['direct_token_f1']}`")

        answer_cols[1].markdown("#### RAG Answer")
        answer_cols[1].write(selected_row["rag_answer"])
        answer_cols[1].write(f"Label: `{selected_row['rag_correctness_label']}`")
        answer_cols[1].write(f"Token F1: `{selected_row['rag_token_f1']}`")

        st.markdown("#### Retrieval and Groundedness")
        grounded_row = groundedness_df[groundedness_df["question_id"].astype(str) == selected_question_id]
        grounded_record = grounded_row.iloc[0].to_dict() if not grounded_row.empty else {}

        retrieval_cols = st.columns(4)
        retrieval_cols[0].metric("BM25 Gold Support", str(selected_row["bm25_gold_supported_in_top_k"]))
        retrieval_cols[1].metric("Gold Support Rank", str(selected_row["bm25_supported_rank"]))
        retrieval_cols[2].metric(
            "Answer Supported",
            str(grounded_record.get("answer_supported_by_retrieved_evidence", "N/A")),
        )
        retrieval_cols[3].metric("Groundedness Bucket", str(grounded_record.get("groundedness_bucket", "N/A")))

        review_row = manual_review_df[manual_review_df["question_id"].astype(str) == selected_question_id]
        if not review_row.empty:
            review_record = review_row.iloc[0]
            st.markdown("#### Manual Review")
            st.write(f"Manual label: `{review_record['manual_label']}`")
            st.write(f"Error type: `{review_record['error_type']}`")
            st.write(review_record["review_notes"])

        if not rag_run_df.empty and "evidence_passages_json" in rag_run_df.columns:
            rag_record = rag_run_df[rag_run_df["question_id"].astype(str) == selected_question_id]
            if not rag_record.empty:
                passages = parse_evidence_passages(rag_record.iloc[0]["evidence_passages_json"])
                st.markdown("#### Retrieved Evidence")
                for index, passage in enumerate(passages, start=1):
                    with st.expander(f"Passage {index}"):
                        st.write(passage)

with manual_tab:
    st.subheader("Manual Review")
    manual_chart = count_chart(
        manual_review_summary.get("error_type_counts", {}),
        "Manual Review Error Types",
        "Rows",
        "Error Type",
    )
    if manual_chart is not None:
        st.plotly_chart(manual_chart, use_container_width=True)

    label_chart = count_chart(
        manual_review_summary.get("manual_label_counts", {}),
        "Manual Review Labels",
        "Rows",
        "Manual Label",
    )
    if label_chart is not None:
        st.plotly_chart(label_chart, use_container_width=True)

    st.dataframe(manual_review_df, use_container_width=True)

with groundedness_tab:
    st.subheader("Groundedness Proxy")
    st.write(
        "This is a lexical evidence-overlap proxy. It is useful for inspection, but it is "
        "not a full entailment or factuality judgment."
    )

    bucket_chart = count_chart(
        groundedness_summary.get("groundedness_bucket_counts", {}),
        "Groundedness Buckets",
        "Rows",
        "Bucket",
    )
    if bucket_chart is not None:
        st.plotly_chart(bucket_chart, use_container_width=True)

    st.dataframe(groundedness_df, use_container_width=True)
