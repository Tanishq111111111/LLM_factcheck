# Pilot 100 Results Summary

## Main Result

Retrieval grounding produced a modest reliability gain on the 100-question TriviaQA pilot benchmark.

| System | Normalized EM | Mean Token F1 | Support@5 |
|---|---:|---:|---:|
| Direct LLM | 0.660 | 0.784 | |
| BM25 Retrieval | | | 0.900 |
| BM25 + RAG | 0.690 | 0.803 | |

## Interpretation

- RAG improved normalized exact match by +0.030 (+3.0 percentage points).
- RAG improved mean token F1 by +0.019.
- BM25 retrieved answer-supporting evidence in 90.0% of benchmark rows.
- RAG fixed 10 direct failures and improved 2 additional cases to partial correctness.
- RAG regressed 9 cases, showing that evidence retrieval alone does not guarantee better answer generation.
- 41 rows were flagged for manual review before final metric or prompt changes.

## Groundedness Proxy

- Gold answer support@5 was 90.0%, so retrieval usually found relevant evidence.
- The RAG answer string appeared in retrieved evidence for 79.0% of all rows.
- Among non-empty, non-abstention RAG answers, lexical answer support was 84.0%.
- Unsupported-answer proxy rate was 15.0%.
- RAG abstained despite top-k gold support in 4 rows.
- Groundedness buckets: supported_correct: 59, unsupported_answer_proxy: 15, supported_partial: 14, supported_but_incorrect: 6, abstained_despite_gold_support: 4, abstained_without_gold_support: 2.

## Manual Review

- Final reviewed rows: 41 of 41.
- Most common reviewed error type: `wrong_granularity` (16 rows).
- Manual labels: both_wrong: 10, ambiguous_partial: 9, direct_partial_rag_correct: 6, direct_correct_rag_wrong: 5, direct_correct_rag_partial: 4, direct_wrong_rag_correct: 4, direct_wrong_rag_partial: 2, metric_artifact: 1.

## Current Conclusion

The project now has evidence that retrieval grounding can improve factual QA reliability, but the improvement is not automatic. The strongest failure pattern is not simple lack of evidence: BM25 often retrieves useful evidence, while RAG can still select the wrong entity, over-answer, or refuse despite support. The next step is to turn these findings into final figures and a concise report narrative.
