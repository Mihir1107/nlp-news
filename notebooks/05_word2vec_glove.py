# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 · Word2Vec and GloVe for Text Classification (Module III)
#
# **Theory.** Word embeddings map each word to a dense vector (here 100 numbers) so that words used in
# similar contexts end up close together. **Word2Vec** learns them with a shallow neural network that
# predicts context: *CBOW* predicts a word from its surrounding window, *skip-gram* predicts the
# surrounding words from the centre word (trained with negative sampling). **GloVe** instead builds a
# global word–word co-occurrence matrix and factorises it, so that `wi·wj + bi + bj ≈ log X_ij`
# (X_ij = how often j appears near i). To classify a document we average its word vectors. This is
# cheap, but **averaging throws away word order** ("dog bites man" = "man bites dog").

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import src.utils  # FIRST: sets cache folders before gensim.downloader / transformers are imported

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from src.utils import FAST, MODELS, SEED
from src.data import AG_LABELS, load
from src.preprocess import clean
from src.evaluate import classification_metrics, plot_confusion, save_results

print("FAST mode:", FAST)

# %% [markdown]
# ## Tokens
# `clean(text, lemma=False)`: lowercase, NLTK tokens, punctuation and stop words removed. We do **not**
# lemmatize, because GloVe's vocabulary holds surface forms ("said", "companies").

# %%
train, test = load("ag_train"), load("ag_test")
t0 = time.perf_counter()
train_tok = [clean(t, lemma=False).split() for t in train.text]
test_tok = [clean(t, lemma=False).split() for t in test.text]
tok_secs = time.perf_counter() - t0
print(f"tokenized {len(train_tok) + len(test_tok):,} docs in {tok_secs:.1f}s")
print(train_tok[0])

# %% [markdown]
# ## Train Word2Vec on the AG training text: CBOW vs skip-gram
# Only the 30k *training* documents are used, so the test set stays unseen. `workers=1` plus
# `PYTHONHASHSEED=0` (set by `scripts/run_nb.sh`) makes training repeatable.

# %%
W2V_ARGS = dict(vector_size=100, window=5, min_count=2, epochs=10, workers=1, seed=SEED)
w2v, w2v_secs = {}, {}
for name, sg in [("cbow", 0), ("skipgram", 1)]:
    t = time.perf_counter()
    w2v[name] = Word2Vec(sentences=train_tok, sg=sg, **W2V_ARGS).wv  # .wv = the word vectors only
    w2v_secs[name] = time.perf_counter() - t
    print(f"{name:9s} vocab {len(w2v[name]):,}  trained in {w2v_secs[name]:.1f}s")

# %% [markdown]
# If you see `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` above: it is a
# harmless message from gensim's compiled code, not a training failure. When a dot product happens to
# equal exactly −1.0 (Cython's error sentinel), Cython checks for an error, finds none, and prints this
# line anyway. We verified no Python exception is raised, the vectors contain no NaN, and the neighbours
# below are sensible.

# %% [markdown]
# ## Load pre-trained GloVe (Wikipedia + Gigaword, 6B tokens, 400k words)

# %%
t = time.perf_counter()
glove = api.load("glove-wiki-gigaword-100")
glove_load_secs = time.perf_counter() - t
print(f"GloVe: {len(glove):,} words × {glove.vector_size} dims, loaded in {glove_load_secs:.1f}s")

# %% [markdown]
# ## Nearest neighbours: what each space learned

# %%
probe_words = ["oil", "microsoft", "football", "iraq", "profit"]
rows = []
for w in probe_words:
    for name, kv in [("w2v cbow", w2v["cbow"]), ("w2v skip-gram", w2v["skipgram"]), ("glove", glove)]:
        nn = kv.most_similar(w, topn=5) if w in kv else []
        rows.append({"word": w, "model": name, "top-5 neighbours": ", ".join(f"{x} ({s:.2f})" for x, s in nn)})
pd.set_option("display.max_colwidth", 120)
pd.DataFrame(rows).set_index(["word", "model"])

# %% [markdown]
# Our Word2Vec was trained on only ~0.8M tokens of 2004 news, so its neighbours are *topical*
# (words from the same stories). GloVe saw 6B tokens and gives cleaner *semantic* neighbours.
#
# ## One analogy: king − man + woman ≈ ?

# %%
for name, kv in [("w2v cbow", w2v["cbow"]), ("w2v skip-gram", w2v["skipgram"]), ("glove", glove)]:
    if all(w in kv for w in ["king", "man", "woman"]):
        print(f"{name:14s}", kv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
    else:
        print(f"{name:14s} missing words:", [w for w in ["king", "man", "woman"] if w not in kv])
print("glove (news analogy) paris - france + japan:",
      glove.most_similar(positive=["paris", "japan"], negative=["france"], topn=3))

# %% [markdown]
# Analogies need a lot of varied text. In 2004 AG News, "king" is mostly Cambodia's King Sihanouk, so
# our Word2Vec answers with Cambodian names. GloVe (6B tokens) recovers *queen*, and even
# *paris − france + japan ≈ tokyo*.
# Note also *microsoft*: our vectors give its products (windows, xp), while GloVe gives its competitors
# (google, ibm). Small in-domain training captures "appears in the same stories"; big general corpora
# capture "plays the same role".

# %% [markdown]
# ## Document vector = mean of its word vectors (OOV words skipped)

# %%
def doc_vectors(token_lists, kv):
    """Average the vectors of the in-vocabulary words; an all-OOV document gets a zero vector."""
    X = np.zeros((len(token_lists), kv.vector_size), dtype=np.float32)
    n_tok = n_oov = n_empty = 0
    for i, toks in enumerate(token_lists):
        known = [w for w in toks if w in kv.key_to_index]
        n_tok += len(toks); n_oov += len(toks) - len(known)
        if known:
            X[i] = kv[known].mean(axis=0)
        else:
            n_empty += 1
    return X, n_oov / max(n_tok, 1), n_empty

features, oov = {}, {}
for name, kv in [("Word2Vec CBOW", w2v["cbow"]), ("Word2Vec skip-gram", w2v["skipgram"]), ("GloVe", glove)]:
    t = time.perf_counter()
    Xtr, oov_tr, _ = doc_vectors(train_tok, kv)
    Xte, oov_te, empty_te = doc_vectors(test_tok, kv)
    features[name] = (Xtr, Xte, (time.perf_counter() - t) / (len(Xtr) + len(Xte)) * 1000)
    oov[name] = {"train_oov_rate": round(oov_tr, 4), "test_oov_rate": round(oov_te, 4), "test_all_oov_docs": empty_te}
pd.DataFrame(oov).T

# %% [markdown]
# Word2Vec's training OOV rate is only the words seen once (`min_count=2`). On *test* text, any word
# that never appeared in 30k AG documents is lost. GloVe's 400k vocabulary covers more test words.

# %% [markdown]
# ## Logistic regression on the averaged vectors (AG test set, 7,600 docs)

# %%
rows, fitted = [], {}
clean_ms = tok_secs / (len(train_tok) + len(test_tok)) * 1000
for name, (Xtr, Xte, vec_ms) in features.items():
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    t = time.perf_counter(); clf.fit(Xtr, train.label); fit_s = time.perf_counter() - t
    t = time.perf_counter(); pred = clf.predict(Xte); pred_ms = (time.perf_counter() - t) / len(Xte) * 1000
    m = classification_metrics(test.label, pred, AG_LABELS)
    kv = glove if name == "GloVe" else w2v["skipgram" if "skip" in name else "cbow"]
    emb_train_s = 0.0 if name == "GloVe" else w2v_secs["skipgram" if "skip" in name else "cbow"]
    fitted[name] = dict(pred=pred, metrics=m, fit_s=fit_s, emb_train_s=emb_train_s,
                        infer_ms=clean_ms + vec_ms + pred_ms,
                        size_mb=kv.vectors.nbytes / 1e6 + clf.coef_.nbytes / 1e6)
    rows.append({"model": name, "accuracy": m["accuracy"], "macro_P": m["macro_precision"],
                 "macro_R": m["macro_recall"], "macro_F1": m["macro_f1"],
                 "embedding train s": round(emb_train_s, 1), "LR fit s": round(fit_s, 1),
                 "infer ms/doc": round(fitted[name]["infer_ms"], 3),
                 "size MB (vectors)": round(fitted[name]["size_mb"], 1)})
results = pd.DataFrame(rows).set_index("model")
results

# %% [markdown]
# Skip-gram beats CBOW clearly, and ends up level with GloVe even though it has ~3× GloVe's test OOV
# rate: vectors trained on the task's own domain make up for a small corpus. CBOW averages its context
# window, which blurs the rarer, topic-specific words that matter most here.
#
# For reference, TF-IDF + the same logistic regression scored the accuracy printed below (from
# `results/tfidf.json`, nb02). Averaged embeddings compress each document into 100 numbers and lose
# which exact words appeared, so on a topic task with strong keywords, sparse TF-IDF is hard to beat.

# %%
import json
from src.utils import RESULTS
print("TF-IDF uni+bigram accuracy:", json.loads((RESULTS / "tfidf.json").read_text())["metrics"]["accuracy"])

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, name in zip(axes, ["Word2Vec skip-gram", "GloVe"]):
    plot_confusion(test.label, fitted[name]["pred"], AG_LABELS, ax=ax, title=name)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## t-SNE of document vectors (1,000 test documents)
# t-SNE squeezes 100 dimensions into 2 while trying to keep near neighbours near. Only the grouping is
# meaningful, not the axes or the distances between clusters.

# %%
idx = np.random.default_rng(SEED).choice(len(test), size=1000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ["Word2Vec skip-gram", "GloVe"]):
    xy = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(features[name][1][idx])
    for c, lab in enumerate(AG_LABELS):
        mask = test.label.to_numpy()[idx] == c
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, label=lab, alpha=0.7)
    ax.set_title(f"t-SNE of averaged {name} vectors"); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(markerscale=3)
plt.tight_layout(); plt.show()

# %% [markdown]
# Sports forms the tightest cluster (a very distinctive vocabulary). Business and Sci/Tech overlap:
# the same confusion the classifiers make.

# %% [markdown]
# ## Why domain match matters
# The same GloVe algorithm trained on tweets (`glove-twitter-100`, used in nb11) would fit news worse:
# its neighbours reflect how words are used on Twitter. We pick the Wikipedia+Gigaword (newswire) vectors
# for news and the Twitter vectors for tweets. Our own Word2Vec is perfectly in-domain but tiny, which is
# why it has more OOV test words and noisier neighbours.

# %%
if not FAST:
    MODELS.mkdir(exist_ok=True)
    w2v["skipgram"].save(str(MODELS / "w2v_ag_skipgram.kv"))  # static vectors reused in nb07 ("bank")
    print("saved models/w2v_ag_skipgram.kv")

for family, main in [("word2vec", "Word2Vec skip-gram"), ("glove", "GloVe")]:
    f = fitted[main]
    save_results(family, {
        "dataset": "ag_news", "n_train": len(train), "n_test": len(test), "config": main + " mean + LR",
        "metrics": f["metrics"], "train_time_s": round(f["emb_train_s"] + f["fit_s"], 2),
        "infer_ms_per_doc": round(f["infer_ms"], 3), "model_size_mb": round(f["size_mb"], 1),
        "oov": oov[main],
        "variants": results.reset_index().to_dict("records") if family == "word2vec" else [],
        "y_pred": f["pred"],
    })

# %% [markdown]
# ## Viva notes
# **Q1. CBOW vs skip-gram?**
# CBOW predicts the centre word from the average of its context words: faster, and good for frequent
# words. Skip-gram predicts each context word from the centre word: slower, but it gives rare words
# more training signal, so it usually represents them better.
#
# **Q2. Word2Vec vs GloVe?**
# Word2Vec is *predictive*: it slides a window and trains a small network with local context.
# GloVe is *count-based*: it first counts global co-occurrences over the whole corpus, then fits
# vectors whose dot products match the log counts. Both give static vectors, one per word type.
#
# **Q3. Why do averaged embeddings lose information?**
# The mean is the same for any order of the words, so syntax and negation scope are lost, and 30 words
# are squeezed into 100 numbers. Rare but decisive keywords get diluted by the other words, which
# TF-IDF keeps as separate features.
