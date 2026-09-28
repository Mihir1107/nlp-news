# 7-minute live demo

## Before the examiner arrives (10 minutes earlier)

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1                     # never depend on the lab Wi-Fi
uvicorn api.app:app --port 8765             # wait for "Application startup complete" (~15 s)
```

Port 8765 because on this laptop another program already uses 8000. Check with `curl localhost:8765/health`
that it answers `{"status":"ok", ...}`.

- Open in the browser: **http://127.0.0.1:8765/** (the web app), `docs/ARCHITECTURE.md` (rendered), and the
  notebooks 02, 03, 07, 12 with their stored outputs. **Do not re-run nb07 live**: fine-tuning takes ~25 minutes.
- In the web app, run one DistilBART summary once so the 1.2 GB model is already loaded.
- Close other apps: this laptop ran out of memory during development.
- If you changed anything in `frontend/src`, rebuild first: `cd frontend && npm run build`.

## Timeline

| Time | Show | Say (one or two sentences) |
|---|---|---|
| 0:00–0:40 | `docs/ARCHITECTURE.md` diagram 1 | Three datasets, one split script with seed 42, twelve notebooks writing JSON results, one comparison notebook. Every number I show comes from those JSON files. |
| 0:40–1:40 | nb02: toy TF-IDF table and the `np.allclose` cell | TF-IDF written from scratch in 69 lines, with sklearn's smoothed idf `ln((1+n)/(1+df))+1`. It matches sklearn exactly. On AG News TF-IDF beats raw counts for unigrams, and the bootstrap CI shows the gap is real. |
| 1:40–2:30 | nb03: log-probability table, perplexity table | Without smoothing, one unseen pair such as *(taxes, next)* makes the probability 0. Bigram beats unigram, but trigram loses on test (64% of test trigrams are unseen) while its training perplexity keeps falling: sparsity. |
| 2:30–3:10 | nb07: "bank" cosine table and the GPT-2 hallucination cell | BERT gives "bank" different vectors in money and river sentences; Word2Vec has one vector. GPT-2 says the 2004 literature Nobel went to Albert Einstein: fluent and false. |
| 3:10–5:40 | The web app (live) | See the steps below. |
| 5:40–6:30 | Web app "The numbers" section, then nb12's "every model gets wrong" table | DistilBERT is best, about 1.5 points over TF-IDF and significant, at ~40× the size. About half its errors are shared by all six models, often label noise: tennis labelled "World". |
| 6:30–7:00 | – | Limitations: small NER model, extractive-reference bias in ROUGE, the QA reader cannot abstain on some unanswerable questions, latency measured on a memory-starved laptop. |

## Web app steps (3:10–5:40)

1. **Topic.** Click the "Giddy Phelps…" headline, choose *Compare both*, then *Find the topic*. Both models
   say Sports; point at the probability bars and the latency difference (milliseconds vs tens or hundreds).
2. **Summary.** The US budget article is preloaded. Click *Summarize*: TextRank marks the 3 sentences it keeps,
   with its rank written in the margin, and the rule beside each sentence is its PageRank score. Say: "the graph is TF-IDF cosine
   similarity between sentences; PageRank is our own power iteration, tested against networkx."
   Optionally switch to DistilBART and show the generated summary, and read the warning under it.
3. **Names & places.** Click *Children vote Shrek 2 best film*, then *Find names and places*. Click
   *organisation* in the key: "Azkaban" and "Dominic Wood" show up as organisations. Real errors of the small
   spaCy model, which matches our hand check in nb04.
4. **Ask the archive.** Click these questions in the "From the question desk" column, in order:
   - *Which play did Uganda's authorities ban?* → the answer as a pull quote; *Show the source passage* marks it in the text.
   - *summarize article 1744* → labelled **rule-based**: a regular expression, not the model.
   - *Who won the 2018 FIFA World Cup?* → "I don't know"; the red tick on the confidence bar is the tuned threshold.
5. Optional: type only spaces in the Topic box and submit → the API's 422 message appears under the button.
6. Optional, as a closing flourish: click the street lamp beside the nameplate. The night edition spreads out
   from the lamp; click again to put it out. (If the room is dark, start in the night edition.)

## If something breaks

- Page says "API not reachable": the server is not running on the port you opened. Start it as above.
- Page loads but looks unstyled or blank: `frontend/dist` is missing; `cd frontend && npm run build`.
- API will not start: `pytest tests/test_api.py -q` shows which model file is missing. Fall back to Swagger
  (`/docs`), the `curl` commands in `README.md`, or nb10's stored chat outputs.
