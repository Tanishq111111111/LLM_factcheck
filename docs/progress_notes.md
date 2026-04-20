# Project Progress Notes

This file tracks the project at a practical level: goal, completed work, current result, and next move.

## Current Status

- Current phase: early comparison experiments.
- Active dataset: TriviaQA pilot benchmark.
- Current benchmark size: 100 questions.
- Completed systems: direct LLM pilot, BM25 retrieval baseline, BM25-grounded RAG pilot.
- Next priority: create comparison reports, add manual review labels, then scale direct and RAG from 10 questions to 100.

## Step 1: Project Setup

- Goal: create a clean, reproducible project structure instead of scattered notebooks and scripts.
- Achieved: added organized folders for `src`, `configs`, `docs`, `notebooks`, `data`, `results`, and `app`.
- Achieved: configured dependencies in `requirements.txt` and used a local `.venv`.
- Why it matters: the project can now grow into a reproducible data science workflow.
- Next change: keep committing stable checkpoints after each major pipeline milestone.

## Step 2: Dataset Inspection

- Goal: inspect TriviaQA before building any model pipeline.
- Achieved: created and fixed `notebooks/01_dataset_inspection.ipynb`.
- Achieved: confirmed useful fields for questions, answer aliases, search results, and entity pages.
- Why it matters: preprocessing choices are based on the actual dataset schema, not assumptions.
- Next change: add Natural Questions only after the TriviaQA pipeline is stable.

## Step 3: Benchmark Creation

- Goal: build a small but useful benchmark file for experiments.
- Achieved: generated `data/benchmark/triviaqa_pilot_v1.csv`.
- Achieved: benchmark has 100 rows and includes question metadata, answer aliases, answer normalization fields, search contexts, and entity contexts.
- Known issue: some evidence text has encoding noise and very long contexts.
- Next change: clean evidence text more aggressively before larger RAG runs.

## Step 4: Direct LLM Baseline

- Goal: measure how well the model answers factual questions without retrieved evidence.
- Achieved: implemented `run_direct.py` with OpenAI and reference providers.
- Achieved: completed a 10-question OpenAI pilot using `gpt-5.4-mini`.
- Result: normalized exact match was 0.70 and mean token F1 was 0.766667.
- Finding: direct prompting missed at least one time-sensitive or changed-fact question, returning `Minecraft` instead of `Super Mario Bros.`.
- Next change: scale direct baseline to all 100 benchmark questions after comparison reporting is in place.

## Step 5: BM25 Retrieval Baseline

- Goal: test whether the benchmark evidence contains answer-supporting passages.
- Achieved: implemented `run_bm25.py`.
- Achieved: ran BM25 top-5 retrieval over all 100 benchmark rows.
- Result: gold answer support@5 was 0.90.
- Finding: retrieved evidence often contains the answer even when direct LLM answering fails.
- Next change: use BM25 output as the evidence source for RAG and groundedness analysis.

## Step 6: Retrieval-Grounded LLM Pilot

- Goal: test whether retrieved passages improve factual answering.
- Achieved: implemented `run_rag.py` with OpenAI and reference providers.
- Achieved: completed a 10-question RAG pilot using BM25 top-5 passages.
- Result: normalized exact match was 0.70 and mean token F1 was 0.823810.
- Finding: RAG fixed the `Minecraft` miss by answering `Super Mario Bros.`, but introduced one `INSUFFICIENT_EVIDENCE` answer where direct prompting was correct.
- Next change: compare direct and RAG row-by-row, then manually label changed cases.

## Step 7: Evaluation

- Goal: automatically score model outputs against aliases.
- Achieved: implemented exact match, normalized exact match, token F1, correctness labels, and risk labels.
- Current limitation: groundedness and hallucination metrics are listed but not implemented yet.
- Current limitation: some answers are semantically close but evaluated harshly, such as morphology or over-specific answers.
- Next change: add manual review labels before changing automated scoring rules.

## Immediate Next Moves

- Build a comparison report that joins direct, BM25, and RAG outputs.
- Create a manual review CSV for changed or failed cases.
- Use the comparison report to decide whether prompt tuning, retrieval tuning, or metric refinement should happen before scaling to 100 questions.
- After review, run direct and RAG over all 100 questions.
