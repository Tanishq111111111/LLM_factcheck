# LLM FactCheck

LLM FactCheck is a reliability evaluation framework for factual question answering systems. The initial pilot compares:

1. Direct LLM answering
2. Retrieval-grounded LLM answering
3. A simple extractive or search-style baseline

The first milestone is not a full model pipeline. It is a clean, reproducible pilot on one dataset with clear metrics, a decision log, and a benchmark subset that can support later experiments.

## Recommended Setup

Use VS Code as the main environment and Jupyter notebooks inside VS Code for exploration.

- Use `notebooks/` for schema inspection, quick experiments, and visual checks.
- Use `src/` for reusable code that will survive beyond one session.
- Use the VS Code terminal for running scripts and Git commands.

On Windows, a plain `venv` is the simplest place to start unless you already use Conda consistently.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Register the kernel for notebooks.
4. Open the repo in VS Code and select the `.venv` interpreter.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name llm-factcheck
```

## First Build Steps

1. Open `notebooks/01_dataset_inspection.ipynb`.
2. Load `TriviaQA` and inspect the schema.
3. Confirm which fields contain questions, answers, aliases, and evidence.
4. Update `configs/dataset_config.yaml` if the field names need adjustment.
5. Move any useful notebook code into `src/data/`.

## Current Scope

- Start with `TriviaQA` only.
- Use BM25 before FAISS.
- Focus on correctness, groundedness, hallucination, and error labeling.
- Delay dashboard work until the pilot pipeline produces results.

## Project Layout

```text
llm-factcheck/
├── app/
├── configs/
├── data/
├── docs/
├── notebooks/
├── results/
└── src/
```

After dependency install, the first useful commands are:

```powershell
python -m src.pipelines.run_eval
python -m src.pipelines.build_triviaqa_pilot
```

See [docs/scope_memo.md](docs/scope_memo.md) for the initial scope and [docs/experiment_plan.md](docs/experiment_plan.md) for the first implementation sequence.
