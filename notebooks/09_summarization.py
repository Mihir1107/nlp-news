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
# # 09 · Text Summarization (Module V)
#
# **Theory.** *Extractive* summarization copies the most important sentences from the article.
# *Abstractive* summarization writes new sentences, like a human would. **Lead-3** (the first three
# sentences) is a strong baseline for news, which puts key facts first. **TextRank** builds a graph of
# sentences linked by similarity and ranks them with PageRank: a sentence is central if it resembles
# other central sentences. **DistilBART-CNN** is a distilled BART encoder–decoder fine-tuned on
# CNN/DailyMail news to generate summaries. **ROUGE** scores a summary by n-gram overlap with a
# reference: ROUGE-1 = unigrams, ROUGE-2 = bigrams, ROUGE-L = longest common subsequence.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import src.utils  # FIRST: cache folders

import re
import time
import numpy as np
import pandas as pd
import spacy
import torch
import transformers
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import FAST, SEED, get_device
from src.data import load
from src.textrank import lead_summary, split_sentences, textrank_summary
from src.evaluate import save_results

transformers.logging.set_verbosity_error()
pd.set_option("display.max_colwidth", 400)
device = get_device()
print("device:", device, "| FAST:", FAST)

# %% [markdown]
# ## Data: the fixed 100-article sample (seed 42) and its reference summaries
# Input is the article *body* (title removed), because the references were written from the body.

# %%
bbc = load("bbc")
sample = bbc[bbc.summ_sample].reset_index(drop=True)
if FAST:
    sample = sample.head(10)
words = lambda s: len(s.split())
print(f"{len(sample)} articles | body words: median {sample.body.map(words).median():.0f} | "
      f"reference words: median {sample.summary.map(words).median():.0f}")
print(sample.category.value_counts().to_dict())

# %% [markdown]
# ### Are the reference summaries extractive?
# For each reference sentence, check whether it appears word for word in the article.

# %%
norm = lambda t: re.sub(r"\s+", "", t)
copied = [np.mean([norm(s) in norm(body) for s in split_sentences(ref)])
          for body, ref in zip(sample.body, sample.summary)]
print(f"reference sentences copied verbatim from the article: {np.mean(copied):.1%} "
      f"(articles with 100% copied references: {np.mean(np.array(copied) == 1):.0%})")

# %% [markdown]
# **Yes: the references are almost entirely extractive**, i.e. sentences selected from the article.
# ROUGE therefore rewards systems that copy article sentences (Lead-3, TextRank) and penalises an
# abstractive model for paraphrasing, even when the paraphrase is correct. Keep this bias in mind below.

# %% [markdown]
# ## System 1 and 2: Lead-3 and TextRank (from scratch, `src/textrank.py`)

# %%
systems, secs = {}, {}
for name, fn in [("Lead-3", lead_summary), ("TextRank", textrank_summary)]:
    t = time.perf_counter()
    systems[name] = [fn(b, k=3) for b in sample.body]
    secs[name] = (time.perf_counter() - t) / len(sample)
    print(f"{name:8s} {secs[name] * 1000:.1f} ms/article")

# %% [markdown]
# ## System 3: DistilBART-CNN (abstractive), called with `generate()` directly
# Beam search with 4 beams, 40–130 generated tokens, input cut to 1,024 tokens (BART's maximum).
# `no_repeat_ngram_size=3` stops the decoder from repeating any 3-gram.

# %%
name = "sshleifer/distilbart-cnn-12-6"
tok = AutoTokenizer.from_pretrained(name)
bart = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
n_long = sum(len(tok(b).input_ids) > 1024 for b in sample.body)
print(f"articles longer than 1,024 tokens (truncated): {n_long}")

@torch.no_grad()
def bart_summary(text):
    enc = tok(text, truncation=True, max_length=1024, return_tensors="pt").to(device)
    out = bart.generate(**enc, num_beams=4, min_length=40, max_length=130,
                        no_repeat_ngram_size=3, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).strip()

t = time.perf_counter()
systems["DistilBART"] = [bart_summary(b) for b in sample.body]
secs["DistilBART"] = (time.perf_counter() - t) / len(sample)
print(f"DistilBART {secs['DistilBART']:.2f} s/article on {device}")

# %% [markdown]
# ## ROUGE-1 / 2 / L against the references
# `rouge-score` with Porter stemming. We report F1 (main) and also precision and recall, because the
# references are long and the systems' output lengths differ.

# %%
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rows, per_article = [], {}
for sys_name, outs in systems.items():
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(sample.summary, outs)]
    per_article[sys_name] = [s["rouge2"].fmeasure for s in scores]
    row = {"system": sys_name}
    for m in ["rouge1", "rouge2", "rougeL"]:
        row[f"{m} F1"] = np.mean([s[m].fmeasure for s in scores])
    for m in ["rouge1"]:
        row[f"{m} P"] = np.mean([s[m].precision for s in scores])
        row[f"{m} R"] = np.mean([s[m].recall for s in scores])
    row["summary words"] = np.mean([words(o) for o in outs])
    row["seconds/article"] = secs[sys_name]
    rows.append(row)
rouge = pd.DataFrame(rows).set_index("system").round(4)
rouge

# %% [markdown]
# **Reading the table.** TextRank beats Lead-3 by a wide margin, and its ROUGE-1 *precision* is
# ~0.95: almost every word it picks is in the reference. The references are extractive, and their
# sentences are not in article order (example 1 below starts mid-article). A plausible explanation is
# that they were built by a sentence-scoring method not unlike TextRank. That is our hypothesis; the
# dataset card does not say. DistilBART is last on ROUGE, but its summaries are short, fluent and
# mostly correct (see below). Low ROUGE here mostly means "did not copy the reference's sentences".

# %% [markdown]
# ## 3 side-by-side examples

# %%
for i in [0, 1, 2]:
    print(f"=== [{sample.category[i]}] {sample.title[i]}")
    print("REFERENCE :", sample.summary[i][:600], "…\n")
    for sys_name in systems:
        print(f"{sys_name:10s}:", systems[sys_name][i], "\n")

# %% [markdown]
# ## Checking the abstractive output for unsupported content
# An automatic proxy for hallucination: numbers and named entities in the summary that **do not appear
# anywhere in the article**. Extractive systems cannot score above 0 here by construction.

# %%
nlp = spacy.load("en_core_web_sm")
def unsupported(summary, article):
    nums = [n for n in re.findall(r"\d[\d,.]*\d|\d", summary) if n not in article]
    ents = [e.text for e in nlp(summary).ents if e.text not in article]
    return nums + ents

checks = [unsupported(s, b) for s, b in zip(systems["DistilBART"], sample.body)]
flagged = [(i, c) for i, c in enumerate(checks) if c]
print(f"DistilBART summaries with a number/entity not found in the article: {len(flagged)} of {len(sample)}")
for i, c in flagged[:6]:
    print(f"  [{i}] {c}  <- {systems['DistilBART'][i][:160]}…")

# %% [markdown]
# ### Hand review of every flagged summary
# ⚠️ Reviewed by the AI assistant against the full article text; check it yourself before the viva.
# Most flags are string-matching artefacts: a possessive *'s*, an entity span swallowing a tennis score,
# or a paraphrase ("At 23" → "The 23-year-old", "just 1.3%" → "Only 1.3%").

# %%
REVIEW = {
    25: ("unsupported fact", "adds the venue \"St James' Park\" (Newcastle's ground): true in the world, "
                             "but not in the article, which never says where the match is"),
    83: ("factual error", "article: 3.1bn Australian dollars ($2.3bn US). Summary: \"$2.3bn Australian "
                          "dollars\": the model merged the two currencies"),
    72: ("repetition", "\"Green Day, the Black Eyed Peas and the Green Day\": a name duplicated"),
}
ARTEFACT = {22, 29, 31, 32, 36, 41, 43, 48, 64, 68, 69, 89}  # checked: supported by the article
assert set(REVIEW) | ARTEFACT == {i for i, _ in flagged}, "outputs changed: review the flags again"
for i, (kind, why) in REVIEW.items():
    print(f"[{i}] {kind.upper()}: {why}\n     summary: {systems['DistilBART'][i]}\n")
print(f"{len(flagged)} flagged -> {sum(k != 'repetition' for k, _ in REVIEW.values())} real content "
      f"errors, 1 repetition, {len(ARTEFACT)} harmless artefacts")

# %% [markdown]
# Two real errors in 100 summaries is a low rate, but look at their *kind*. The currency merge changes
# the value of a takeover bid and still reads perfectly naturally. The added venue is "world knowledge"
# the model brought from its training data, not from the source. Neither would be caught by ROUGE.
# An extractive system can make neither error.

# %%
save_results("summarization", {
    "dataset": "bbc-news-summary (100-article fixed sample)", "n_articles": len(sample),
    "reference_sentences_copied_pct": round(float(np.mean(copied)) * 100, 1),
    "rouge": rouge.reset_index().to_dict("records"),
    "distilbart": {"num_beams": 4, "min_length": 40, "max_length": 130, "input_max_tokens": 1024,
                   "device": str(device), "n_truncated_inputs": int(n_long),
                   "summaries_with_unsupported_number_or_entity": len(flagged)},
    "examples": [{"title": sample.title[i], **{k: v[i] for k, v in systems.items()}} for i in range(3)],
    "outputs": {"ids": sample.id.tolist(), **systems},       # every summary, for review and nb12
    "unsupported_check": {str(i): c for i, c in flagged},
    "hand_review": {"real_errors": {str(k): v for k, v in REVIEW.items() if v[0] != "repetition"},
                    "repetition": [k for k, v in REVIEW.items() if v[0] == "repetition"],
                    "artefacts": sorted(ARTEFACT), "reviewed_by": "AI assistant; to be checked by student"},
})

# %% [markdown]
# ## Why ROUGE is a flawed judge
# - It counts overlapping words, not meaning: a correct paraphrase scores low and a fluent sentence with
#   one wrong number scores high.
# - With a single reference, anything the reference writer happened not to pick is "wrong".
# - Here the references are extractive and long, which favours extractive systems and long outputs.
#   DistilBART is capped at 130 tokens, so its recall is limited by construction.
#
# ## Viva notes
# **Q1. Extractive vs abstractive summarization?**
# Extractive picks existing sentences, so it is faithful to the source but can be choppy. Abstractive
# generates new text (DistilBART), which is more fluent and compact but can introduce facts that are not
# in the source.
#
# **Q2. How does TextRank work?**
# Sentences become TF-IDF vectors, and cosine similarities form a weighted graph. PageRank (power
# iteration, `d = 0.85`) scores each sentence by how strongly it connects to other high-scoring
# sentences. We keep the top 3 in original order. Our PageRank equals networkx's (unit test).
#
# **Q3. Why is ROUGE flawed here?**
# It measures n-gram overlap with one reference, not correctness or readability. These references
# copy article sentences, so ROUGE is biased toward extractive methods. Lead-3 is hard to beat on
# news because journalists put the key facts first.
