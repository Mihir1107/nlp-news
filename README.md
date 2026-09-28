# NewsMind

NLP-Based News Article Classification, Summarization, and Question Answering System.
Semester VII NLP lab project, GLS University (course 2604734).

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/download_data.py            # datasets, NLTK, spaCy, GloVe, HF models (~4 GB), splits
python scripts/download_data.py --skip-models   # lighter: no transformer weights
pytest                                     # split tests
```

`NEWSMIND_FAST=1` shrinks AG/tweet splits to 500 rows per class for quick runs.
Device is chosen automatically: CUDA > Apple MPS > CPU.

## Layout

```
src/        from-scratch modules + shared helpers (data.py, utils.py, ...)
notebooks/  01-12, jupytext .py sources + executed .ipynb
api/        FastAPI app
tests/      pytest
docs/       research_notes.md (verified versions, datasets, deviations), architecture, demo script
data/       gitignored; recreated by scripts/download_data.py
```

See `docs/research_notes.md` for dataset schemas, model sizes and deviations from the plan.
