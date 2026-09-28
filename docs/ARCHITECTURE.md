# NewsMind architecture

## 1. Data → experiments → results

```mermaid
flowchart LR
    subgraph Sources["Hugging Face / GitHub"]
        AG[AG News]
        BBC[BBC News + summaries]
        TW[TweetEval sentiment]
        EMB[GloVe wiki + twitter]
        HF[Pre-trained models<br/>DistilBERT · DistilBART · BERT<br/>GPT-2 · MiniLM · RoBERTa · SQuAD reader]
    end

    DL[scripts/download_data.py<br/>retries every download] --> RAW[(data/raw/*.parquet)]
    AG & BBC & TW --> DL
    EMB & HF --> DL
    RAW --> SPLIT["src/data.py build_splits()<br/>repair text · dedupe · seed 42"] --> SPL[(data/splits/*.parquet)]

    SPL --> NB["notebooks 01–11<br/>(use src/preprocess, tfidf, ngram,<br/>textrank, qa, evaluate)"]
    NB --> RES[(results/*.json)]
    NB --> MOD[(models/<br/>tfidf_lr.joblib · distilbert-agnews/<br/>w2v_ag_skipgram.kv)]
    NB --> ENT[(data/entities.parquet)]
    RES --> NB12[notebook 12] --> DOCS[docs/RESULTS.md<br/>README table<br/>report/figures]
```

Rule: every number in the docs comes from `results/*.json`. nb12 is the only place that turns them into
tables, and it refuses to run if any result was produced in FAST mode.

## 2. The API at run time

```mermaid
flowchart TB
    W["Web app (frontend/dist, served at /)<br/>React + Framer Motion"] -->|fetch JSON| V
    C[curl / Swagger UI at /docs] -->|JSON| V{"Pydantic validation<br/>1–10,000 chars, known model/method"}
    V -->|invalid| E422[HTTP 422 + reason]
    V --> R{Endpoint}

    R -->|/classify tfidf| T["clean() → TF-IDF → LogisticRegression"]
    R -->|/classify distilbert| D["tokenize, truncate at 512 → fine-tuned DistilBERT"]
    R -->|/summarize textrank| TR["src/textrank.py<br/>TF-IDF sentence graph → PageRank → top-k"]
    R -->|/summarize distilbart| DB["DistilBART generate()<br/>lazy-loaded · input cut at 1,024 tokens"]
    R -->|/entities| S[spaCy en_core_web_sm NER]
    R -->|/results /samples| J["results/*.json · docs/qa_pairs.json<br/>real example texts"]
    R -->|/ask| NBOT

    subgraph NBOT["NewsBot (src/qa.py)"]
        direction TB
        RT{"Rule-based router<br/>(regular expressions)"}
        RT -->|summarize article N| TR2[TextRank on article N]
        RT -->|who is mentioned in article N| EP[PERSON entities from entities.parquet]
        RT -->|what topic is …| CL[TF-IDF classifier]
        RT -->|anything else| RET["Retriever: TF-IDF cosine over 9,937<br/>120-word passages (40-word overlap) → top 3"]
        RET --> RD["Reader: DistilBERT-SQuAD<br/>best span by start+end logit, ≤30 tokens"]
        RD --> TH{"confidence ≥ threshold<br/>(tuned in nb10)?"}
        TH -->|yes| ANS[answer + source article]
        TH -->|no| IDK["I don't know."]
    end
```

Start-up (FastAPI `lifespan`) loads TF-IDF+LR, DistilBERT, spaCy and NewsBot once, then runs one warm-up
prediction per classifier so the first real request is not slow. DistilBART (1.2 GB) loads on its first request.
The built web app is mounted at `/` *after* all API routes, so it can never shadow an endpoint.

## 3. The web app (`frontend/`)

One page with five views, each calling one endpoint. Every model result is drawn onto the text itself with
an animated highlighter mark (`components/ui.jsx:Marker`, a background gradient whose width grows 0 → 100%).

| View | Endpoint | What it shows |
|---|---|---|
| Topic | `POST /classify` | label, class probabilities as animated bars; "Compare both" runs TF-IDF and DistilBERT side by side |
| Summary | `POST /summarize` | TextRank: the whole article, kept sentences marked in rank order, PageRank score per sentence. DistilBART: generated text revealed word by word |
| Names and places | `POST /entities` | entities marked in the text by kind; the legend filters one kind |
| Ask the archive | `POST /ask` | chat; answer marked inside its source passage; confidence bar with the tuned "I don't know" threshold |
| How good is it? | `GET /results` | dot plot of classifier accuracy, bars for ROUGE, QA and tweet results, all read from `results/*.json` |

Build: `esbuild` bundles `src/main.jsx` (JSX, CSS, fonts) into `dist/` (`frontend/build.mjs`). Fonts are bundled,
so the app needs no internet. Motion respects the operating system's "reduce motion" setting.

## 4. Modules

| File | Role | From scratch? | Tested in |
|---|---|---|---|
| `src/utils.py` | paths, cache folders, seed, device (CUDA > MPS > CPU), timer, FAST flag | – | used everywhere |
| `src/data.py` | download → splits; AG/BBC/tweet text repair; dedup | – | `tests/test_splits.py` |
| `src/preprocess.py` | `clean(text, mode="news" \| "tweet")` and each step on its own | yes (uses NLTK pieces) | `tests/test_preprocess.py` |
| `src/tfidf.py` | BoW and TF-IDF, identical to sklearn's defaults | **yes** | `tests/test_tfidf.py` (allclose vs sklearn) |
| `src/ngram.py` | n-gram LM: MLE / Laplace / add-k, `<UNK>`, perplexity, generation | **yes** | `tests/test_ngram.py` (hand-computed probabilities) |
| `src/textrank.py` | TextRank + PageRank power iteration | **yes** | `tests/test_textrank.py` (PageRank vs networkx) |
| `src/qa.py` | chunking, retriever, reader, NewsBot, router | retriever uses our TF-IDF | `tests/test_qa.py` |
| `src/evaluate.py` | accuracy, macro P/R/F1, confusion matrix, SQuAD EM/F1, results JSON | metrics via sklearn, EM/F1 ours | `tests/test_qa.py` |
| `api/app.py` | FastAPI service, also serves the web app | – | `tests/test_api.py` |
| `frontend/src/` | React web app (App.jsx, one component per view, ui.jsx shared pieces) | – | checked in a browser (see README) |
