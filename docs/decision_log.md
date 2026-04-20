# Decision Log

## 2026-03-25

- Decided to start with one dataset only before combining datasets.
- Decided to use TriviaQA as the first pilot dataset.
- Decided to use BM25 before trying FAISS.
- Decided to separate raw, interim, processed, and benchmark data.
- Decided to keep notebooks for exploration and move reusable code into `src/`.
- Decided to use `requirements.txt` as the dependency source of truth for now.
- Decided to delay Streamlit work until the pilot pipeline produces evaluation outputs.
