# Metric Definitions

- Exact Match: predicted answer exactly matches a gold answer string.
- Normalized Exact Match: compare prediction and gold after normalization.
- Token F1: overlap-based score between predicted and gold tokens.
- Semantic Similarity: embedding-based similarity between prediction and gold answer.
- Groundedness: whether the predicted answer is supported by retrieved evidence.
- Groundedness Proxy: lexical check for whether the RAG answer string appears in the retrieved top-k passages. This is conservative and can miss paraphrases.
- Unsupported Answer Proxy: a non-empty, non-abstention RAG answer that does not lexically appear in the retrieved evidence.
- Supported But Incorrect: the predicted answer appears in retrieved evidence but does not match the gold answer, suggesting evidence-selection or reasoning failure rather than pure hallucination.
- Hallucination: answer includes unsupported or fabricated details. In this pilot, hallucination is approximated with the unsupported answer proxy and manual review labels.
- Correctness Label: correct, partially correct, or incorrect.
- Risk Flag: low, medium, or high risk based on correctness and support.
