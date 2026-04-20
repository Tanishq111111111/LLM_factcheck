# LLM FactCheck

LLM FactCheck is a reliability evaluation framework for factual question answering systems. The pilot compares:

1. Direct LLM answering
2. BM25 retrieval as a search-style evidence baseline
3. BM25-grounded RAG answering

The project is designed as a reproducible data science workflow: build a benchmark subset, run comparable systems, evaluate correctness, inspect grounding, and manually review failure modes.

## Current Status

- Dataset: TriviaQA pilot benchmark with 100 questions.
- Completed systems: direct OpenAI baseline, BM25 top-5 retrieval, and BM25-grounded OpenAI RAG.
- Completed analysis: direct-vs-RAG comparison, human-verified manual review, and lexical groundedness proxy.
- Main pilot result: RAG improved normalized exact match from 0.66 to 0.69 and mean token F1 from 0.784048 to 0.803024.

## Recommended Setup

Use VS Code as the main environment and Jupyter notebooks inside VS Code for exploration.

- Use `notebooks/` for schema inspection, quick experiments, and visual checks.
- Use `src/` for reusable code that should survive beyond one session.
- Use the VS Code terminal for running scripts and Git commands.

On Windows, a plain `venv` is the simplest place to start unless you already use Conda consistently.

## Quick Start

Run these commands from the repo root in `cmd`:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name llm-factcheck
```

## Useful Commands

```cmd
.venv\Scripts\python.exe -m src.pipelines.build_triviaqa_pilot
.venv\Scripts\python.exe -m src.pipelines.run_bm25
.venv\Scripts\python.exe -m src.pipelines.run_eval
.venv\Scripts\python.exe -m src.pipelines.compare_runs
.venv\Scripts\python.exe -m src.pipelines.score_groundedness
.venv\Scripts\python.exe -m src.pipelines.build_results_summary
.venv\Scripts\python.exe -m src.pipelines.build_figures
```

OpenAI runs also require setting `OPENAI_API_KEY` in the same `cmd` window before running `run_direct.py` or `run_rag.py`.

```cmd
set OPENAI_API_KEY=your_actual_api_key_here
```

## Dashboard

After the pilot result files exist, run the Streamlit dashboard from the repo root:

```cmd
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

## Project Layout

```text
llm-factcheck/
|-- app/
|-- configs/
|-- data/
|-- docs/
|-- notebooks/
|-- results/
`-- src/
```

See `docs/progress_notes.md` for the current project state, `docs/decision_log.md` for major choices, `results/tables/pilot100_results_summary.md` for the latest report-ready results, and `docs/final_report_draft.md` for the current written report draft.
