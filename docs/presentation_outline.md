# Presentation Outline

## Slide 1: Title

LLM FactCheck: Building a Reliability Evaluation Framework for Factual Question Answering

Team: Baavansh Reddy Gundlapalli and Tanishq Annavaram

## Slide 2: Motivation

- Users increasingly treat LLMs like search engines.
- A fluent answer can still be incomplete, unsupported, outdated, or hallucinated.
- The project evaluates when direct LLM answers are reliable and when retrieval grounding helps.

Main takeaway: accuracy alone is not enough; we also need evidence support and failure analysis.

## Slide 3: Research Question

Under what conditions can an LLM be trusted for factual question answering, and how much does retrieval grounding improve reliability compared with direct prompting alone?

## Slide 4: Dataset and Benchmark

- Dataset: TriviaQA
- Pilot benchmark: 100 factual QA rows
- Fields used: question text, gold answer, answer aliases, search/entity evidence contexts
- Why 100 rows first: enough to validate the full pipeline while keeping manual review feasible

## Slide 5: System Design

Compare three systems:

- Direct LLM: answers from model knowledge only
- BM25 Retrieval: retrieves top-5 evidence passages
- BM25 + RAG: answers using only retrieved evidence

Pipeline:

Benchmark -> Direct Run -> BM25 Retrieval -> RAG Run -> Evaluation -> Comparison -> Manual Review -> Groundedness Analysis

## Slide 6: Metrics

Correctness:

- Exact Match
- Normalized Exact Match
- Token F1
- Correct / Partially Correct / Incorrect labels

Reliability:

- BM25 gold support@5
- RAG answer evidence-overlap proxy
- Unsupported-answer proxy
- Refusal despite gold support
- Manual error taxonomy

## Slide 7: Main Results

Use figure: `results/figures/pilot100_metric_comparison.png`

Key numbers:

- Direct normalized EM: 0.66
- RAG normalized EM: 0.69
- Direct mean token F1: 0.784
- RAG mean token F1: 0.803
- BM25 gold support@5: 0.90

Interpretation:

RAG improved results, but the improvement was modest.

## Slide 8: Row-Level Comparison

- RAG fixed 10 direct failures.
- RAG improved 2 additional cases to partial correctness.
- RAG regressed 9 cases.
- 41 rows were flagged for manual review.

Main point:

Retrieval can help, but it can also introduce new failure modes.

## Slide 9: Groundedness Analysis

Use figure: `results/figures/pilot100_groundedness_buckets.png`

Key numbers:

- RAG answer support rate across all rows: 79.0%
- RAG answer support rate among answered rows: 84.0%
- Unsupported-answer proxy rate: 15.0%
- RAG refused despite gold support in 4 rows.
- 6 rows were supported by retrieved evidence but still incorrect.

Main point:

Retrieval found relevant evidence often, but answer selection and abstention still failed in some cases.

## Slide 10: Manual Error Analysis

Use figure: `results/figures/pilot100_manual_review_error_types.png`

Top error types:

- Wrong granularity: 16
- Incorrect entity: 9
- Evidence use failure: 7
- Over-complete answer: 4

Main point:

The most common issues were not just hallucinations; many failures involved answer format, specificity, or poor use of retrieved evidence.

## Slide 11: What We Learned

- Retrieval grounding modestly improves factual QA reliability.
- BM25 retrieved gold-supporting evidence in 90% of rows.
- The generation step remains a bottleneck.
- Manual review is necessary because automatic metrics can misclassify ambiguous or differently phrased answers.
- A reliability framework should include correctness, evidence support, abstention behavior, and interpretable error categories.

## Slide 12: Limitations

- Pilot benchmark has only 100 questions.
- Groundedness metric is lexical, not full entailment.
- Only TriviaQA is used so far.
- Only one main LLM configuration is tested.
- BM25 is a baseline retriever, not the strongest possible retrieval system.

## Slide 13: Future Work

- Scale benchmark size.
- Add Natural Questions or a domain-specific dataset.
- Test dense or hybrid retrieval.
- Add entailment-based groundedness scoring.
- Build a Streamlit dashboard for row-level inspection.
- Compare multiple models and prompt strategies.

## Slide 14: Final Takeaway

Retrieval improves reliability, but it does not solve factual QA by itself. The system must also evaluate whether retrieved evidence is used correctly, whether answers are properly scoped, and whether refusals are justified.

