# 7-minute live demo

## Before the examiner arrives (10 minutes earlier)

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1                     # never depend on the lab Wi-Fi
uvicorn api.app:app                         # wait for "Application startup complete" (~15 s)
```

- Open in the browser: `http://127.0.0.1:8000/docs`, `docs/ARCHITECTURE.md` (rendered), and the notebooks
  02, 03, 07, 12 with their stored outputs. **Do not re-run nb07 live**: fine-tuning takes ~25 minutes.
- Send one `/summarize` request with `"method": "distilbart"` once, so the 1.2 GB model is already loaded.
- Close other apps: this laptop ran out of memory during development.

## Timeline

| Time | Show | Say (one or two sentences) |
|---|---|---|
| 0:00–0:40 | `docs/ARCHITECTURE.md` diagram 1 | Three datasets, one split script with seed 42, twelve notebooks writing JSON results, one comparison notebook. Every number I show comes from those JSON files. |
| 0:40–1:40 | nb02: toy TF-IDF table and the `np.allclose` cell | TF-IDF written from scratch in 69 lines, with sklearn's smoothed idf `ln((1+n)/(1+df))+1`. It matches sklearn exactly. On AG News TF-IDF beats raw counts for unigrams, and the bootstrap CI shows the gap is real. |
| 1:40–2:40 | nb03: log-probability table, perplexity table | Without smoothing, one unseen pair such as *(taxes, next)* makes the probability 0. Bigram beats unigram, but trigram loses on test (64% of test trigrams are unseen) while its training perplexity keeps falling: sparsity. |
| 2:40–3:30 | nb07: "bank" cosine table and the GPT-2 hallucination cell | BERT gives "bank" different vectors in money and river sentences; Word2Vec has one vector. GPT-2 says the 2004 literature Nobel went to Albert Einstein: fluent and false. |
| 3:30–5:30 | Swagger UI `/docs` (live) | See the API steps below. |
| 5:30–6:30 | nb12: Table 1, bootstrap table, "every model gets wrong" examples | DistilBERT is best (~1.5 points over TF-IDF, significant), at ~40× the size. About half its errors are shared by all six models, often label noise: tennis labelled "World". |
| 6:30–7:00 | – | Limitations: small NER model, extractive-reference bias in ROUGE, the QA reader cannot abstain on some unanswerable questions, latency measured on a memory-starved laptop. |

## API steps (3:30–5:30), in Swagger UI "Try it out"

1. `POST /classify` `{"text": "Apple unveiled a faster iPod and new iMac computers at its annual conference."}`:
   Sci/Tech with TF-IDF (default, ~1 ms). Repeat with `"model": "distilbert"`: same label, higher latency.
2. `POST /classify` `{"text": ""}`: **422**. Point at the validation rule in `api/app.py`.
3. `POST /summarize` with a pasted BBC paragraph, `textrank`, then `distilbart`.
4. `POST /ask` in this order:
   - `"Which play did Uganda ban?"` → *The Vagina Monologues*, with the source article.
   - `"summarize article 1744"` → the **rule-based** router (a regex), not the model.
   - `"who is mentioned in article 109"` → people from the NER file.
   - `"Who won the 2018 FIFA World Cup?"` → *I don't know* (confidence below the tuned threshold).

## If something breaks

- API will not start: `pytest tests/test_api.py -q` shows which model file is missing. Fall back to nb10's
  stored chat demo outputs.
- Browser problem: use the `curl` commands in `README.md`.
