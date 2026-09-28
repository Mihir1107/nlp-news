# Research notes (Phase 0 verification)

Verified 2026-09-28 on MacBook Air M3, 16 GB, macOS (Darwin 27). Every number below came
from a command run in this repo. Deviations from the original plan are marked **DEVIATION**.

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12.13 (uv-managed venv in `.venv/`). System default is 3.14; 3.12 chosen for wheel safety |
| transformers | 4.57.6 (`transformers<5` pin resolves) |
| torch | 2.14.0, `torch.backends.mps.is_available()` = True, `get_device()` returns `mps` |
| numpy / scipy | 2.5.3 / 1.18.1, gensim 4.4.0 imports and trains (`FAST_VERSION=1`, compiled C path) |
| datasets | 5.0.1 |
| sentence-transformers | 5.7.0 |
| spaCy | 3.8.16 + `en_core_web_sm` 3.8.0 |
| NLTK | 3.10.3 |
| scikit-learn | 1.9.1 |
| Full pins | `requirements.txt` (direct), `requirements-lock.txt` (all transitive) |

Environment variables set in `src/utils.py` (import it first, before transformers or gensim):
`HF_HOME=data/hf_cache`, `GENSIM_DATA_DIR=data/gensim_data`, `NLTK_DATA=data/nltk_data`,
`PYTORCH_ENABLE_MPS_FALLBACK=1`, `TOKENIZERS_PARALLELISM=false`,
`DISABLE_SAFETENSORS_CONVERSION=true` (see §5). `NEWSMIND_FAST=1` keeps 500 rows per class
for AG and tweet splits (AG train 2,000 / AG test 2,000 / tweet train 1,500).

**Disk:** the laptop had 11 GB free at the start. After setup: `.venv` 1.3 GB, HF models 3.2 GB,
GloVe 516 MB, NLTK 58 MB, raw + splits 57 MB. **About 5.9 GB free now.** The fine-tuned
DistilBERT (~260 MB) and notebook outputs will fit, but nothing large should be added.

**Network:** the local DNS resolver dropped out repeatedly during setup (`nodename nor servname
provided`). `scripts/download_data.py` retries every network step (6 tries, 15 s apart).
After download, all models load with `HF_HUB_OFFLINE=1`, so notebooks do not need the network.

## 2. NLTK resources

All five names in the plan are correct for NLTK 3.10: `punkt_tab`, `stopwords`, `wordnet`,
`omw-1.4`, `averaged_perceptron_tagger_eng`. Stored in `data/nltk_data/`.
Note: the English stopword list now has **198** words (older tutorials say 179).

## 3. Datasets

### AG News: `fancyzhx/ag_news` ✅ matches plan
- 120,000 train / 7,600 test; columns `text` (string), `label` (ClassLabel: World, Sports, Business, Sci/Tech).
- Perfectly balanced: 30,000 per class in train, 1,900 per class in test.
- **Raw-text artefacts** (counts in the 120k train set): `#39;` (the apostrophe) in 29,696 rows,
  `quot;` in 4,746, `&lt;…&gt;` HTML tags in 5,241, `amp;` in 1,467, and literal backslashes
  (escaped newlines, e.g. `dwindling\band`). `src/data.py:repair_ag_text()` fixes these.
  The original is kept in column `text_raw`.
- No exact raw-text overlap between train and test. **But after repair**, 18 train rows become
  identical to test rows and 166 train rows duplicate each other. These are removed from the
  train pool before sampling. The test set is untouched (full 7,600).
- Length: median 37 words, 95th percentile 52 words. So `max_len=64` wordpiece tokens will
  truncate a small tail of rows (BERT splits words into roughly 1.3 pieces each).

### BBC News: `SetFit/bbc-news` ❌ **DEVIATION: not used**
- 1,225 train / 1,000 test (2,225 total); columns `text`, `label`, `label_text`.
- The text is **lowercased and apostrophes are stripped** (`"wales want rugby league training
  wales could follow england s lead"`). That breaks spaCy NER and POS (capital letters are a
  main NER cue), sentence splitting for TextRank, and any summarization comparison.
- **Fix applied (please confirm):** use `gopalkalpande/bbc-news-summary` as the single BBC source
  for nb03, nb04, nb08, nb09 and nb10. It holds the same corpus in original case, with the
  category in `File_path` and a reference summary for every article. The category counts match
  SetFit's to within one article. SetFit is still downloaded to `data/raw/` but is not used.

### BBC summaries: `gopalkalpande/bbc-news-summary` ✅ loads (CC0)
- 2,224 rows, one split `train`; columns `File_path` (category), `Articles`, `Summaries`.
- Categories: sport 510, business 510, politics 417, tech 401, entertainment 386.
- **Formatting damage:** newlines were turned into `.` when the dataset was converted. The title is
  joined to the body by `..`, paragraphs are glued (`funds...Duncan`), and summary sentences
  have no space between them (`£50,000.Mr Bannatyne`). `parse_bbc_article()` and
  `fix_glued_sentences()` repair this. They leave abbreviations like `U.S.` alone (unit-tested).
- **Duplicates:** 98 articles appear twice verbatim, and 37 more reuse the same body under a
  reworded title (e.g. "Labour accused of 'EU propaganda'" / "Labour's 'EU propaganda'").
  All are dropped, leaving **2,089 articles**: business 500, sport 491, politics 395,
  entertainment 366, tech 337.
- **The reference summaries are extractive.** 96.1% of summary sentences appear word-for-word
  in the article (76.1% of summaries are 100% copied). The remaining "misses" are sentence-splitting
  artefacts (bullet lists, quotes), not paraphrases. 95.6% of summary tokens appear in the article.
  → nb09 must state that ROUGE against these references favours extractive methods (Lead-3, TextRank).
- **Summaries are long:** median 148 words against a 325-word article body (about 43% of the article).
  DistilBART with `max_length≈130` *tokens* produces shorter output, so ROUGE recall is capped for
  it. Report P/R alongside F1 in nb09, or note this.
- Body length: median 325 words, max 4,430. DistilBART truncation to 1,024 tokens affects only long articles.

### TweetEval sentiment: `cardiffnlp/tweet_eval`, config `sentiment` ✅ loads directly
- Loads under `datasets` 5.0.1 with no script error (the Hub serves the parquet conversion
  automatically). No GitHub fallback was needed.
- 45,615 train / 2,000 validation / 12,284 test; `label` ClassLabel (negative, neutral, positive).
- **Label shift:** negatives are 15.5% of train, 15.6% of validation, but **32.3% of test**.
  Positives: 39.1% train, 19.3% test. This is why TweetEval's official metric is macro-recall,
  and it will lower accuracy of any model trained on the train distribution.
- Mentions are already anonymised to `@user`.

## 4. Splits (`data/splits/`, seed 42, built by `src/data.py:build_splits()`)

| File | Rows | Notes |
|---|---|---|
| ag_train | 30,000 | exactly 7,500 per class, shuffled |
| ag_train_20k | 20,000 | 5,000 per class, subset of ag_train (DistilBERT) |
| ag_val | 2,000 | 500 per class, disjoint from ag_train (for training curves only) |
| ag_test | 7,600 | full official test set |
| bbc | 2,089 | `id, category, title, body, summary, label, lm_split, summ_sample` |
| tweet_train/val/test | 45,615 / 2,000 / 12,284 | official splits |

- `bbc.lm_split`: 80/20 **article-level** split stratified by category, for the n-gram LM (nb03).
  Splitting by article rather than by sentence stops sentences from the same article ending up in both sides.
- `bbc.summ_sample`: the fixed 100 articles for ROUGE (nb09).
- `ag_val` was not in the original plan. It is 2,000 rows held out so nb06/nb07 can watch
  training without touching the test set. It is never used for tuning.
- `tests/test_splits.py`: 9 tests (sizes, balance, disjointness, subset nesting, artefacts removed,
  BBC dedup, rebuild is byte-identical, FAST mode, repair functions on hand examples). **All pass.**

## 5. Models (Hub, verified 2026-09-28)

| Planned id | Resolves to | Weights | Size |
|---|---|---|---|
| distilbert-base-uncased | distilbert/distilbert-base-uncased | safetensors | 268 MB |
| sshleifer/distilbart-cnn-12-6 | (same) | **pytorch_model.bin only** | 1,222 MB |
| distilbert-base-cased-distilled-squad | distilbert/distilbert-base-cased-distilled-squad | safetensors | 261 MB |
| gpt2 | openai-community/gpt2 | safetensors | 548 MB |
| bert-base-uncased | google-bert/bert-base-uncased | safetensors | 440 MB |
| sentence-transformers/all-MiniLM-L6-v2 | (same) | safetensors | 91 MB |
| cardiffnlp/twitter-roberta-base-sentiment-latest | (same) | **pytorch_model.bin only** | 501 MB |

- We use the namespaced ids in code (`scripts/download_data.py:HF_MODELS`).
- **Gotcha:** for the two `.bin`-only models, transformers 4.57 starts a background thread that
  downloads a safetensors conversion from a Hub pull request. That meant a second full copy (+1.7 GB).
  Fixed with `DISABLE_SAFETENSORS_CONVERSION=true` in `src/utils.py`, and the duplicates were deleted.
- Offline smoke test (every model through the exact API the notebooks will use, on MPS):
  - DistilBART `generate(num_beams=4)`: coherent summary, 4.8 s including load.
  - QA `AutoModelForQuestionAnswering` argmax span: "Who will deliver the Budget?" → "Gordon Brown".
  - BERT fill-mask: "Paris is the [MASK] of France." → capital; "The central bank raised interest
    [MASK] to fight inflation." → rates. (Unused pooler/NSP weights warning is expected for MaskedLM.)
  - GPT-2 greedy: works, with `pad_token_id=eos_token_id`.
  - twitter-roberta: `id2label = {0: negative, 1: neutral, 2: positive}`, same order as TweetEval.
  - MiniLM: cos("stocks fell", "shares dropped") = 0.72, cos("stocks fell", "football match") = 0.03.
- `pipeline("summarization")` and `pipeline("question-answering")` are not used anywhere (plan requirement).

## 6. Embeddings (gensim.downloader, cached in `data/gensim_data/`)

| Name | Download | Vocab | Notes |
|---|---|---|---|
| glove-wiki-gigaword-100 | 128 MB | 400,000 | uncased, Wikipedia 2014 + Gigaword 5 |
| glove-twitter-100 | 406 MB (listed) | 1,193,514 | uncased, 2B tweets |

- `most_similar("economy")` → economic 0.83, growth 0.79, recession 0.77.
- Loading both takes about 70 s, mostly the Twitter model. Budget for this in nb11.
- gensim.downloader fetches its loader code from GitHub (accepted for coursework).

## 7. Decisions (approved at the Phase 0 checkpoint: "do what you think is best")
1. BBC source is `gopalkalpande/bbc-news-summary`, deduplicated to 2,089 articles. SetFit is not used.
2. `ag_val` (2,000 rows) is kept, only for monitoring training in nb06 and nb07.

## 8. Found during Phase 1
- **TweetEval escapes.** 11.2% of train tweets contain literal escape text (`it’s`), and emoji
  are stored as escaped UTF-16 surrogate pairs. `repair_tweet_text()` decodes them (original kept in `text_raw`).
- **TweetEval train has zero emoji; 6.53% of test tweets have emoji.** A model cannot learn emoji
  features from train. Mapping emoji to words (😂 → "laugh") is the only way they can help (nb11 ablation).
- **More AG entities.** Bare `hellip;`, `nbsp;`, `rsquo;`, `mdash;`, `lt;`, `gt;` (about 70 rows) are now
  repaired. Fixing them changed the dedup pool, so the seeded 30k train sample changed. nb01 and nb02 were
  re-run afterwards, and all saved numbers are from the final data.
- **pandas 3 + pyarrow strings.** `.str.contains()` uses the RE2 engine, which has **no backreferences**
  (`(\w)\1` fails). Use Python `re` through `.map()` for those patterns.
- **WordNet lemmatizer quirk.** It strips a final "s" whenever the rest is a dictionary word: `us→u`,
  `has→ha`, `boss→bos`. `src/preprocess.py` guards against this (unit-tested).
- **`NEWSMIND_FAST=1`** writes results to `results/fast/` (gitignored), so it never replaces real numbers.

## 9. Found during Phase 2
- **Import order matters.** `import src.utils` must come before `gensim.downloader`, otherwise GloVe
  downloads again into `~/gensim-data` (it happened once; the stray copy was deleted). Every notebook from
  nb05 on imports `src.utils` first.
- **gensim prints `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'`** during skip-gram
  training (gensim 4.4.0, numpy 2.5.3). No Python exception is raised (`sys.unraisablehook` never fires),
  the vectors have no NaN and the neighbours are sensible. It is a spurious Cython sentinel check.
- **Word2Vec reproducibility** needs `workers=1` and `PYTHONHASHSEED=0` (set in `scripts/run_nb.sh`).
  Two runs gave identical accuracy (0.8861).
- **MPS is slower than CPU for the small packed BiLSTM** (nb06 measures both each run; it chose CPU).
- **DistilBERT fine-tuning on MPS took 24.1 min** (1,250 steps, ~1 s/step) while the laptop used 7–9 GB
  of swap. `Trainer` on MPS works with `save_strategy="no"` (no checkpoints, to save disk).
- **Gotcha:** a predict-only `Trainer` reusing training args with `eval_strategy="epoch"` raises
  `ValueError` (no eval_dataset). nb07 uses separate prediction args.
- **Latency numbers are noisy on this machine.** DistilBERT batched inference measured 44.6 ms/doc in one
  run and 10.0 ms/doc in the next, same model and data, depending on memory pressure. Treat ms/doc as
  order-of-magnitude, and say so in nb12 and the viva.

## 10. Found during Phase 3
- **BBC HTML entities.** 22 articles contained entities like `Jos&#233;` (found when QA answered "EC President
  Jos&#233"). `parse_bbc_article()` now calls `html.unescape`. All BBC notebooks (nb03, nb04, nb08, nb09,
  nb10) were re-run afterwards; the numbers barely moved (e.g. trigram PPL 2891.8 → 2891.0, ROUGE identical).
- **Reference summaries.** 96.1% of reference sentences are copied from the article, and they are not in
  article order. TextRank reaches ROUGE-1 precision ~0.95. Hypothesis (not confirmed by the dataset card):
  the references were produced by an extractive sentence-scoring method.
- **DistilBART on MPS:** ~2.7 s per article (100 articles ≈ 5 min). 4 of 100 inputs exceeded 1,024 tokens.
- **QA reader calibration.** In 5 of 7 end-to-end errors the correct article was retrieved, but the reader
  preferred a span from a different passage. Start/end logits are not comparable across passages. The SQuAD
  1.1 reader also gives >0.99 confidence to some answers for unanswerable questions, which caps what a
  threshold can do.
- **TweetEval emoticons:** 3.1% of train tweets but only 0.4% (44) of test tweets contain `:)`-style
  emoticons. The small, significant TF-IDF gain from "emoji handling" is a training-side effect (verified on
  test tweets with no emoji/emoticon: +0.0032 macro-F1, CI [+0.0007, +0.0054]).
- **Test tweets with hashtags: 39.3%** (not ~19%, which was an early train-set estimate).

## 11. Found while building the web app
- **npm registry stalls for some packages on this network.** Metadata for `vite`, `framer-motion`,
  `react-dom` and `scheduler` repeatedly stalled mid-download (e.g. 283 KB after 60 s), while others (`react`,
  `esbuild`) came back in under a second. Clearing the npm cache and using the npmmirror.com mirror did not help.
  - Vite was dropped: `esbuild` alone bundles the app (`frontend/build.mjs`), which is also simpler to explain.
  - React 19.2.8, react-dom, scheduler, framer-motion 12.42.2 (+ motion-dom, motion-utils, tslib) and
    esbuild 0.25.12 (+ its macOS arm64 binary) were **copied from other projects already on this laptop**
    into `frontend/node_modules`. The two font packages were downloaded as tarballs; their SHA-1 matched the
    registry's published checksums. `frontend/package.json` pins exactly these versions, so on a normal
    network `npm install` gives the same tree.
- **Port 8000 was already used by another program on this laptop.** The demo script starts the API on 8765.
- **Browser check (Playwright, Chromium):** all five views exercised at 1360×900 and 390×844 (no horizontal
  scrolling), keyboard tab navigation with arrow keys, visible focus ring, the 422 error path, the DistilBART
  lazy load (3.0 s including load), and no console errors other than the intended 422. Screenshots are in
  `report/figures/web_*.png`.
