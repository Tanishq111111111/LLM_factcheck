# Project Progress Notes

This file tracks the project at a practical level: goal, completed work, current result, and next move.

## Current Status

- Current phase: post-review analysis and final artifact preparation.
- Active dataset: TriviaQA pilot benchmark.
- Current benchmark size: 100 questions.
- Completed systems: direct LLM baseline, BM25 retrieval baseline, BM25-grounded RAG baseline.
- Next priority: prepare final visual/report artifacts from the completed 100-question analysis.

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

## Step 8: Pilot Comparison and Manual Review

- Goal: compare direct, BM25, and RAG outputs in one reproducible table.
- Achieved: generated `pilot10_direct_vs_rag_comparison.csv`.
- Achieved: generated and labeled `pilot10_manual_review_candidates.csv`.
- Result: RAG fixed 2 cases, regressed 2 cases, and left 1 ambiguous partial case.
- Finding: the first RAG prompt sometimes refused despite retrieved evidence and sometimes returned over-complete answers.
- Finding: a tuned prompt was tested but reduced normalized EM from 0.70 to 0.50 and lost the Super Mario Bros. fix.
- Next change: keep the original RAG prompt for scaling, and treat prompt v2 as a failed tuning experiment.

## Step 9: 100-Question Direct vs RAG Comparison

- Goal: test whether retrieval grounding improves reliability beyond the 10-question smoke run.
- Achieved: completed direct and RAG runs over all 100 TriviaQA pilot questions.
- Direct result: normalized exact match 0.66 and mean token F1 0.784048.
- RAG result: normalized exact match 0.69 and mean token F1 0.803024.
- BM25 result: gold support@5 remained 0.90 across the 100-question benchmark.
- Finding: RAG modestly improved aggregate performance, fixing 10 cases and improving 2 partial cases, but regressed 9 cases.
- Achieved: manually validated the 41 flagged comparison rows using the pre-labeled review draft.
- Review result: most review cases involve wrong granularity, incorrect entities, evidence-use failure, or over-complete answers.
- Next change: use the manual review summary to support final error analysis and decide groundedness/hallucination scoring.

## Step 10: Groundedness and Reliability Analysis

- Goal: move beyond answer accuracy and estimate whether RAG answers are supported by retrieved evidence.
- Achieved: finalized the 41-row manual review file as human verified.
- Achieved: generated a RAG groundedness proxy over all 100 pilot rows.
- Result: RAG answer support by lexical evidence overlap was 0.79 across all rows and 0.840426 across answered rows.
- Result: unsupported-answer proxy rate was 0.15.
- Finding: 6 rows were supported by retrieved evidence but still incorrect, showing that retrieval can surface the right context while generation still chooses the wrong answer.
- Finding: RAG refused in 4 rows even though BM25 had top-k gold support.
- Next change: use these outputs to build final figures and the written results narrative.

## Step 11: Report-Ready Artifacts

- Goal: convert experiment outputs into tables and figures that can be used in the final report or presentation.
- Achieved: regenerated `pilot100_results_summary.md` with direct, BM25, RAG, manual-review, and groundedness findings.
- Achieved: generated figures for metric comparison, manual-review error types, and groundedness buckets.
- Next change: write the final narrative explaining when retrieval helped, when it failed, and what the limitations are.

## Immediate Next Moves

- Add a short results narrative for the final report.
- Consider a small dashboard only after final static figures are ready.
