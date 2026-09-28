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
# # 10 · Question Answering Chatbot: NewsBot (Module V)
#
# **Theory.** *Extractive QA* answers a question by selecting a span of text from a document. Because
# a reader model can only read a few hundred tokens at a time, open-domain QA is done in two stages:
# a **retriever** finds a few candidate passages (here TF-IDF cosine over ~120-word chunks), then a
# **reader** (DistilBERT fine-tuned on SQuAD) predicts, for every token, how likely it is to be the
# answer's *start* and *end*. The best span is the one with the highest start + end score. A confidence
# threshold lets the bot say "I don't know" instead of guessing.
#
# **What is rule-based:** the *intent router* (`NewsBot.chat`) is plain regular expressions ("summarize
# article N", "who is mentioned in article N", "what topic is …"). It does not learn anything. Only the
# retriever statistics and the reader are data-driven.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import src.utils  # FIRST: cache folders

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers

from src.utils import DATA, MODELS, ROOT, get_device
from src.data import load
from src.qa import NewsBot, chunk_articles
from src.evaluate import exact_match, save_results, token_f1

transformers.logging.set_verbosity_error()
pd.set_option("display.max_colwidth", 120)

# %% [markdown]
# ## Build NewsBot on all 2,089 BBC articles

# %%
bbc = load("bbc")
t = time.perf_counter()
bot = NewsBot(bbc, entities=pd.read_parquet(DATA / "entities.parquet"),
              classifier=joblib.load(MODELS / "tfidf_lr.joblib"), device=get_device(), threshold=0.2)
print(f"start-up {time.perf_counter() - t:.1f}s | {len(bot.retriever.passages):,} passages "
      f"(120 words, 40-word overlap) | TF-IDF vocab {len(bot.retriever.vec.vocabulary_):,} | "
      f"reader on {bot.reader.device}")

# %% [markdown]
# ## Chat demo: one message per intent

# %%
demo = ["Who did Rafael Nadal beat in the Davis Cup final?",
        "summarize article 1744",
        "who is mentioned in article 109",
        "what topic is article 1012",
        "what topic is this: The striker scored twice as United won the derby.",
        "Who won the 2018 FIFA World Cup?"]
for msg in demo:
    r = bot.chat(msg)
    extra = f"  (confidence {r['confidence']}, article {r['article_id']} [{r['category']}] {r['title']})" \
        if r["intent"] == "question" else ""
    print(f"> {msg}\n  [{r['intent']}] {r['answer']}{extra}\n")

# %% [markdown]
# ## Evaluation on 40 hand-written questions
# `docs/qa_pairs.json`: 30 answerable questions drafted from real BBC passages (answers copied verbatim)
# plus 10 questions the corpus cannot answer. ⚠️ Drafted by the AI assistant, so review them.
#
# Metrics: **hit@k** = the correct article is among the top-k retrieved passages. **EM** = exact match
# after SQuAD normalisation. **F1** = word-overlap F1 with the best acceptable answer.

# %%
qa = json.loads((ROOT / "docs" / "qa_pairs.json").read_text())
rows = []
for item in qa["answerable"]:
    r = bot.ask(item["question"])
    rows.append({"question": item["question"], "gold": item["answers"][0], "pred": r["best_guess"],
                 "confidence": r["confidence"], "EM": exact_match(r["best_guess"], item["answers"]),
                 "F1": token_f1(r["best_guess"], item["answers"]),
                 "hit@1": r["retrieved_article_ids"][0] == item["article_id"],
                 "hit@3": item["article_id"] in r["retrieved_article_ids"], "answerable": True})
for q in qa["unanswerable"]:
    r = bot.ask(q)
    rows.append({"question": q, "gold": None, "pred": r["best_guess"], "confidence": r["confidence"],
                 "answerable": False})
ev = pd.DataFrame(rows)
ans = ev[ev.answerable]
print(f"retrieval hit@1 {ans['hit@1'].mean():.3f}  hit@3 {ans['hit@3'].mean():.3f}  | "
      f"end-to-end reader EM {ans.EM.mean():.3f}  F1 {ans.F1.mean():.3f}  (n={len(ans)})")
ans.drop(columns=["answerable"])

# %% [markdown]
# ### Reader alone: given the correct passage
# To separate the two stages, we also give the reader the gold passage directly (the passage of the
# correct article that contains the answer). The gap between this and end-to-end = errors caused by retrieval.

# %%
passages = chunk_articles(bbc)
oracle = []
for item in qa["answerable"]:
    gold_p = next(p["text"] for p in passages
                  if p["article_id"] == item["article_id"] and item["answers"][0] in p["text"])
    span = bot.reader.best_span(item["question"], [gold_p])
    oracle.append({"EM": exact_match(span["answer"], item["answers"]),
                   "F1": token_f1(span["answer"], item["answers"]), "pred": span["answer"]})
oracle = pd.DataFrame(oracle)
print(f"reader with gold passage: EM {oracle.EM.mean():.3f}  F1 {oracle.F1.mean():.3f}")

# %% [markdown]
# ### Errors, one by one

# %%
wrong = ans[ans.F1 < 1].assign(oracle_pred=oracle.pred[ans.F1.values < 1].values)
wrong[["question", "gold", "pred", "oracle_pred", "confidence", "F1", "hit@3"]]

# %% [markdown]
# **What the errors show.**
# - **Most errors are distractor passages, not retrieval misses.** In every wrong case the right article
#   *was* in the top 3 (hit@3 = True), and the reader alone gets 5 of the 7 right when given the gold
#   passage ("0.8%" instead of "0.5%", "Dubai Open" instead of "French Open"). Start/end logits are not
#   calibrated across passages, so a confident-looking span in a similar article can win. A fix would be
#   to re-rank by retrieval score plus reader score, or pass only the top-1 passage when it is clearly best.
# - **Span boundary:** "80,000 barrels a day *to 84 million barrels a day*" is too long (partial F1).
# - **Reader error:** "Who is Pakistan's High Commissioner?" → "General Musharraf", even from the gold
#   passage: the reader picks the most prominent name.
# - Also visible in the demo: "who is mentioned in article 109" lists *Labour* as a person, the spaCy
#   NER error already seen in nb04.

# %% [markdown]
# ## Tuning the "I don't know" threshold
# A question is handled **correctly** when (answerable) the bot answers with F1 ≥ 0.5, or (unanswerable)
# the bot says "I don't know". We try every threshold and keep the best. Caveat: we tune and measure
# on the same 40 questions, so the chosen threshold's score is optimistic.

# %%
conf = ev.confidence.to_numpy()
good_answer = (ev.answerable & (ev.F1.fillna(0) >= 0.5)).to_numpy()
unanswerable = (~ev.answerable).to_numpy()
grid = np.unique(np.concatenate([[0.0], conf, [1.01]]))
acc = [np.mean(np.where(conf >= t, good_answer, unanswerable)) for t in grid]
best = int(np.argmax(acc))
# several thresholds can tie; take the midpoint between the best one and the next confidence value
threshold = float((grid[best] + grid[min(best + 1, len(grid) - 1)]) / 2) if best > 0 else 0.0
print(f"best threshold ≈ {threshold:.3f} -> {acc[best]:.1%} of the 40 questions handled correctly "
      f"(threshold 0: {acc[0]:.1%})")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(grid, acc, drawstyle="steps-post")
ax.axvline(threshold, ls="--", c="gray")
ax.set_xlabel("confidence threshold"); ax.set_ylabel("share handled correctly"); ax.set_xscale("symlog", linthresh=0.01)
ax.set_title("Answer-or-abstain accuracy vs threshold (40 questions)")
plt.tight_layout(); plt.show()

print("confidence, answerable questions  : median", round(float(np.median(conf[~unanswerable])), 3))
print("confidence, unanswerable questions: median", round(float(np.median(conf[unanswerable])), 3))
ev[~ev.answerable][["question", "pred", "confidence"]]

# %% [markdown]
# For an unanswerable question the reader still returns its best span from whatever was retrieved,
# because it is forced to pick *something*. The reader was trained on SQuAD 1.1, which has no
# unanswerable questions, and its confidence is relative *within one passage*. So some nonsense answers
# get very high confidence ("Who directed Oppenheimer?" → a different director, "Who founded OpenAI?" →
# Khodorkovsky). **No threshold can catch those**, which is why the best threshold handles only part
# of the unanswerable set. A reader trained on SQuAD 2.0 (which has a "no answer" option) would do better.

# %%
bot.threshold = threshold
for msg in ["Who won the 2018 FIFA World Cup?", "Who is the head of Tiga?"]:
    r = bot.chat(msg)
    print(f"> {msg}\n  {r['answer']}  (confidence {r['confidence']})")

save_results("qa", {
    "n_passages": len(passages), "chunk_words": 120, "overlap_words": 40, "retriever": "TF-IDF (src/tfidf.py)",
    "reader": "distilbert-base-cased-distilled-squad", "n_answerable": len(ans), "n_unanswerable": int(unanswerable.sum()),
    "retrieval_hit_at_1": round(float(ans["hit@1"].mean()), 3), "retrieval_hit_at_3": round(float(ans["hit@3"].mean()), 3),
    "reader_em": round(float(ans.EM.mean()), 3), "reader_f1": round(float(ans.F1.mean()), 3),
    "oracle_reader_em": round(float(oracle.EM.mean()), 3), "oracle_reader_f1": round(float(oracle.F1.mean()), 3),
    "threshold": round(threshold, 4), "answer_or_abstain_accuracy": round(float(acc[best]), 3),
    "questions": ev.drop(columns=[]).to_dict("records"),
})

# %% [markdown]
# ## Viva notes
# **Q1. Which part of the chatbot is rule-based?**
# The intent router in `NewsBot.chat`: regular expressions for "summarize article N", "who is mentioned
# in article N" and "what topic is …". Everything else goes to the QA pipeline. The QA part is
# data-driven: TF-IDF statistics for retrieval and a SQuAD-trained DistilBERT reader.
#
# **Q2. How is the answer span chosen?**
# For each of the top-3 passages the reader outputs a start logit and an end logit per token. We score
# every span (i, j) with j ≥ i and length ≤ 30 tokens as start[i] + end[j], and take the best over all
# passages. Confidence = softmax(start)[i] × softmax(end)[j] within that passage.
#
# **Q3. Why chunk articles with overlap?**
# The reader has a 384-token window, and short passages make retrieval more precise. With a 40-word
# overlap, an answer that falls on a chunk boundary is still whole in one of the two chunks.
