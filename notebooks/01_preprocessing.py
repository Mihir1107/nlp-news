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
# # 01 · Text Preprocessing (Module I)
#
# **Theory.** Raw text is a string of characters; models need a sequence of units (tokens).
# *Tokenization* splits text into tokens. *Normalisation* makes different surface forms count as one
# feature: lowercasing, removing punctuation and very common *stop words*, and reducing words to a
# base form. *Stemming* chops suffixes with rules ("studies" → "studi"); *lemmatization* looks the word
# up in a dictionary using its part of speech ("studies"/VERB → "study"). Every step shrinks the
# vocabulary, but each one can also throw away meaning, so the choice depends on the task.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # lets us "import src..." from notebooks/

import time
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from src.utils import FAST
from src.data import load
from src import preprocess as pp
from src.evaluate import save_results

pd.set_option("display.max_colwidth", 140)
nlp = spacy.load("en_core_web_sm")
print("FAST mode:", FAST)

# %% [markdown]
# ## 5 sample AG News headlines + descriptions
# One per class, plus one extra with numbers and an abbreviation.

# %%
ag = load("ag_train")
samples = pd.concat([ag.groupby("label").head(1), ag[ag.text.str.contains(r"\$\d")].head(1)])
samples = samples[["label_name", "text"]].reset_index(drop=True)
samples

# %% [markdown]
# ## Step 1 · Lowercasing
# "Stocks" and "stocks" become one feature. Cost: we lose the cue that "Apple" (company) differs from "apple".

# %%
pd.DataFrame({"before": samples.text, "after": samples.text.map(pp.lowercase)})

# %% [markdown]
# ## Step 2 · Tokenization: NLTK vs regex vs spaCy
# The three tokenizers disagree on contractions, currency, abbreviations and hyphens.

# %%
tricky = "The U.S. firm AT&T didn't pay $10.5m to e-mail users in N.Y. -- shares fell 3%."
tok = {
    "NLTK word_tokenize": pp.tokenize(tricky),
    "regex [A-Za-z0-9]+": pp.regex_tokenize(tricky),
    "spaCy": [t.text for t in nlp(tricky)],
}
for name, toks in tok.items():
    print(f"{name:20s} ({len(toks):2d}) {toks}")

# %% [markdown]
# - NLTK splits *didn't* → *did* + *n't* (Penn Treebank convention) and keeps *U.S.* whole.
# - The regex drops every symbol, so *$*, *%* and *&* vanish and *AT&T* becomes *AT*, *T*.
# - spaCy splits *e-mail* into *e*, *-*, *mail* and *$10.5m* into *$*, *10.5*, *m*.
#
# On the 5 samples (token counts per tokenizer):

# %%
pd.DataFrame({
    "nltk": samples.text.map(lambda t: len(pp.tokenize(t))),
    "regex": samples.text.map(lambda t: len(pp.regex_tokenize(t))),
    "spacy": samples.text.map(lambda t: len(nlp(t))),
    "first 12 NLTK tokens": samples.text.map(lambda t: pp.tokenize(t.lower())[:12]),
})

# %% [markdown]
# ## Step 3 · Stop-word removal
# NLTK's English list has 198 words. What gets removed from each sample:

# %%
nltk_stop = set(stopwords.words("english"))
print("NLTK stop words:", len(nltk_stop), "| ours (negations kept):", len(pp.STOP_WORDS))
rows = []
for text in samples.text:
    toks = pp.tokenize(text.lower())
    rows.append({"removed": [t for t in toks if t in pp.STOP_WORDS],
                 "kept": pp.remove_stopwords(toks)})
pd.DataFrame(rows)

# %% [markdown]
# **Where stop-word removal hurts.** The full NLTK list contains *not*, *no*, *nor* and *don't*, so
# naive removal flips the meaning of a sentence. That is why `src/preprocess.py` keeps negations.

# %%
for s in ["The movie was not good.", "No deal was reached.", "This is not a bad phone."]:
    toks = pp.tokenize(s.lower())
    print(f"{s:28s} naive: {pp.remove_stopwords(toks, nltk_stop)!s:28s} ours: {pp.remove_stopwords(toks)}")

# %% [markdown]
# ## Step 4 · Punctuation removal
# Drop tokens made only of punctuation. Words containing punctuation (*n't*, *u.s.*) survive.

# %%
pd.DataFrame([{"before": pp.tokenize(t.lower()), "after": pp.remove_punctuation(pp.tokenize(t.lower()))}
              for t in samples.text])

# %% [markdown]
# ## Step 5 · Stemming vs lemmatization (10 words)
# The POS column is given by hand, so WordNet gets the right part of speech. "WordNet (no POS)"
# treats every word as a noun, which is its default.

# %%
words = [("running", "v"), ("ran", "v"), ("studies", "n"), ("studying", "v"), ("better", "a"),
         ("wolves", "n"), ("happily", "r"), ("organization", "n"), ("universities", "n"), ("was", "v")]
porter, snowball, wnl = PorterStemmer(), SnowballStemmer("english"), WordNetLemmatizer()
pd.DataFrame([{
    "word": w, "POS": p,
    "Porter": porter.stem(w), "Snowball": snowball.stem(w),
    "WordNet (no POS)": wnl.lemmatize(w), "WordNet (+POS)": wnl.lemmatize(w, p),
    "spaCy": nlp(w)[0].lemma_,
} for w, p in words])

# %% [markdown]
# - Stemmers produce non-words (*studi*, *happili*, *organ*). Porter maps *organization* → *organ*,
#   merging two unrelated meanings.
# - Lemmatizers produce real words, but only if they know the POS: without it, *ran* stays *ran*
#   and *better* stays *better*. With POS, *better*/ADJ → *good* and *was* → *be*.
# - spaCy tags the single word itself, so its guess can differ from our hand-given POS.
#
# **A WordNet trap we found in the AG data.** WordNet strips a final *s* whenever the remainder is
# in its dictionary, so *US* (the country) becomes *u*. `src/preprocess.py` undoes this for very short
# results and for words ending in *ss*:

# %%
pd.DataFrame([{"word": w, "WordNet": wnl.lemmatize(w), "our lemmatize()": pp.lemmatize([w])[0]}
              for w in ["us", "has", "boss", "cars", "news"]])

# %% [markdown]
# ## Full pipeline: `clean(text, mode="news")`

# %%
pd.DataFrame({"before": samples.text, "clean()": samples.text.map(pp.clean)})

# %% [markdown]
# ## Tweet mode: `clean(text, mode="tweet")`
# Extra steps before the news pipeline: URLs → `url`, @mentions → `user`, hashtags split on
# camel case (`#GameDay` → `game day`), known emoji/emoticons → words (😂 → `laugh`), other emoji
# removed, and elongations shortened (`soooo` → `soo`).

# %%
tw = pd.concat([load("tweet_test"), load("tweet_train")])
picks = [
    tw[tw.text.str.contains("😂")].iloc[0],
    tw[tw.text.str.contains("😍")].iloc[0],
    tw[tw.text.str.contains(r"#[A-Z][a-z]+[A-Z]")].iloc[0],
    tw[tw.text.map(lambda t: bool(pp.re.search(r"([a-z])\1{3,}", t)))].iloc[0],
    tw[tw.text.str.contains(r"http|:\)")].iloc[0],
]
tweets = pd.DataFrame(picks)[["label_name", "text"]].reset_index(drop=True)
tweets["clean(tweet)"] = tweets.text.map(lambda t: pp.clean(t, mode="tweet"))
tweets["clean(news)"] = tweets.text.map(pp.clean)
tweets

# %% [markdown]
# News mode on tweets turns `#GameDay` into `gameday`, keeps `@` fragments and drops emoji,
# which removes most of the sentiment signal in an emoji-only tweet.

# %% [markdown]
# ## Vocabulary size after each step (AG News train)
# Each row adds one step on top of the previous one. The last row replaces lemmatization with Porter stemming.

# %%
docs = ag.text.tolist()
t0 = time.perf_counter()
steps = {}
steps["raw (split on spaces)"] = [d.split() for d in docs]
steps["+ lowercase + NLTK tokenize"] = [pp.tokenize(d.lower()) for d in docs]
steps["+ remove punctuation"] = [pp.remove_punctuation(t) for t in steps["+ lowercase + NLTK tokenize"]]
steps["+ remove stop words"] = [pp.remove_stopwords(t) for t in steps["+ remove punctuation"]]
steps["+ lemmatize (WordNet+POS)"] = [pp.lemmatize(t) for t in steps["+ remove stop words"]]
steps["(alt) + Porter stem instead"] = [[porter.stem(w) for w in t] for t in steps["+ remove stop words"]]
secs = time.perf_counter() - t0

vocab = pd.DataFrame([{"step": k, "vocab size": len({w for doc in v for w in doc}),
                       "tokens": sum(len(doc) for doc in v)} for k, v in steps.items()])
vocab["vocab vs raw"] = (vocab["vocab size"] / vocab["vocab size"][0]).map("{:.1%}".format)
print(f"{len(docs):,} documents processed in {secs:.1f}s")
vocab

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(vocab.step[::-1], vocab["vocab size"][::-1], color="#4C78A8")
for i, v in enumerate(vocab["vocab size"][::-1]):
    ax.text(v, i, f" {v:,}", va="center", fontsize=9)
ax.set_xlabel("distinct tokens"); ax.set_title("Vocabulary size after each preprocessing step (AG train)")
ax.set_xlim(0, vocab["vocab size"].max() * 1.15)
plt.tight_layout(); plt.show()

# %%
save_results("preprocess", {
    "dataset": "ag_train", "n_docs": len(docs), "fast_mode": FAST, "seconds": round(secs, 1),
    "vocab_by_step": dict(zip(vocab.step, vocab["vocab size"].tolist())),
    "tokens_by_step": dict(zip(vocab.step, vocab["tokens"].tolist())),
})

# %% [markdown]
# ## Viva notes
# **Q1. Stemming vs lemmatization?**
# Stemming cuts suffixes with fixed rules and can output non-words (*studies* → *studi*). Lemmatization
# returns the dictionary form using the POS tag (*was*/VERB → *be*), so it is slower but more accurate.
#
# **Q2. Why keep "not" when removing stop words?**
# NLTK's list includes negations, and removing them flips meaning ("not good" → "good"). For
# classification and sentiment we keep *no, not, nor, never, n't* (see `src/preprocess.py`).
#
# **Q3. Why does the lemmatizer need POS tags?**
# WordNet assumes a noun by default, so *ran* and *better* are left unchanged. Given VERB or ADJ it
# returns *run* and *good*. We get POS tags from NLTK's perceptron tagger (`pp.lemmatize`).
