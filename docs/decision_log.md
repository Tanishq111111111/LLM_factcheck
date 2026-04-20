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
- Labeled pilot manual-review cases and identified two RAG prompt issues: unsupported refusal and over-complete answers.
- Updated the RAG prompt to prefer one shortest evidence-supported answer and reserve `INSUFFICIENT_EVIDENCE` for genuinely unsupported cases.
- Tested the tuned RAG prompt on 10 questions; it performed worse than the original RAG prompt, so the project reverted to the original RAG prompt before scaling.
- Completed 100-question direct and RAG runs; RAG improved normalized EM from 0.66 to 0.69 and mean token F1 from 0.784048 to 0.803024.
- Decided to manually review the 41 flagged 100-question comparison rows before modifying metrics or prompts further.
- Human-validated the 41-row pre-labeled manual review draft and accepted the generated labels as accurate.
- Added a lexical groundedness proxy for RAG outputs, while recording its limitation that lexical overlap is not equivalent to full factual support.
- Added static report figures before dashboard work so the final deliverable has stable, presentation-ready artifacts first.
- Added a lightweight Streamlit dashboard after completing static figures and summaries, so the UI reads finalized artifacts instead of driving the experiment workflow.
