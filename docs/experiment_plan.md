# Experiment Plan

## Phase 1: Setup and Inspection

1. Inspect TriviaQA schema and confirm usable fields.
2. Define the pilot subset size and sampling rule.
3. Lock the first metric set and error labels.
4. Keep all reusable logic in `src/`.

## Phase 2: Benchmark Preparation

1. Normalize answer aliases.
2. Build a clean benchmark file in `data/benchmark/`.
3. Record all filtering and sampling decisions in the decision log.

## Phase 3: Baselines

1. Add a direct-answering client.
2. Add a BM25 retrieval baseline.
3. Add a retrieval-grounded generation path.

## Phase 4: Evaluation

1. Compute exact match, normalized exact match, and token F1.
2. Add semantic similarity if needed after the base metrics work.
3. Add groundedness and hallucination labels.

## Working Rule

If code is useful beyond one notebook session, move it into `src/` immediately.
