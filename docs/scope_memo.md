# Scope Memo

## Project Goal

Build a reliability evaluation framework for factual question answering systems.

## Core Comparison

We will compare:

1. Direct LLM answering
2. Retrieval-grounded LLM answering
3. A simple extractive or search-style baseline

## Main Question

Under what conditions can an LLM be trusted for factual QA, and how much does retrieval grounding improve reliability?

## Initial Scope

We will start with one dataset first, using TriviaQA as the pilot benchmark. After the pilot is stable, we can decide whether Natural Questions adds value or just more preprocessing noise.

## Success Criteria

- A clean pilot benchmark subset
- Reproducible data loading and preprocessing
- Clear metric definitions
- A usable error taxonomy
- Direct vs retrieval-grounded comparisons with interpretable evidence

## Not In Scope Yet

- FAISS or advanced vector indexing
- A polished dashboard
- Multi-dataset benchmarking
- Domain-specific extensions such as health or finance
