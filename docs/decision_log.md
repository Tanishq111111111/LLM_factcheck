# Decision Log

## 2026-03-25

- Decided to start with one dataset only before combining datasets.
- Decided to use TriviaQA as the first pilot dataset.
- Decided to use BM25 before trying FAISS.
- Decided to separate raw, interim, processed, and benchmark data.
- Decided to keep notebooks for exploration and move reusable code into `src/`.
- Decided to use `requirements.txt` as the dependency source of truth for now.
- Decided to delay Streamlit work until the pilot pipeline produces evaluation outputs.

## 2026-04-20

- Completed a 10-question direct OpenAI pilot with `gpt-5.4-mini`.
- Added retry logic for incomplete OpenAI responses and raised retry output budget to avoid blank predictions.
- Completed a BM25 top-5 retrieval baseline over the 100-question TriviaQA pilot.
- Completed a 10-question BM25-grounded RAG pilot using `gpt-5.4-mini`.
- Decided to add comparison and manual-review reports before scaling direct and RAG to all 100 rows.
