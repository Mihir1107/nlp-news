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
# # 08 · Syntactic and Semantic Analysis with Deep Learning (Module IV)
#
# **Theory.** *Syntactic* analysis recovers sentence structure. A **dependency parse** links every word
# to its *head* with a labelled relation (`nsubj` = subject, `dobj` = direct object, `prep`/`pobj` =
# preposition and its object). spaCy's parser is a neural transition-based parser: a CNN encodes the
# words and a classifier picks shift/arc actions. From the tree we can read *who did what to whom*
# as (subject, verb, object) triples. *Semantic* analysis compares meaning. A **sentence-transformer**
# (all-MiniLM-L6-v2, a 6-layer Transformer fine-tuned on sentence pairs) maps a text to a 384-dim vector,
# so that texts with similar meaning are close even when they share no words. TF-IDF can only match
# the exact words.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import src.utils  # FIRST: cache folders

import time
import nltk
import numpy as np
import pandas as pd
import spacy
from spacy import displacy
from sentence_transformers import SentenceTransformer

from src.utils import FAST, SEED, get_device
from src.data import load
from src.preprocess import clean
from src.tfidf import TfidfVectorizer
from src.evaluate import save_results

pd.set_option("display.max_colwidth", 160)
nlp = spacy.load("en_core_web_sm")
bbc = load("bbc")

# %% [markdown]
# ## Part 1 · Dependency parsing and (subject, verb, object) triples
# First sentences of 20 BBC articles (4 per category, seed 42).

# %%
picked = bbc.groupby("category").sample(4, random_state=SEED)
sentences = [nltk.sent_tokenize(b)[0] for b in picked.body]

def phrase(tok, chunks):
    """The noun chunk containing tok ("the new factory"), or just the word itself."""
    return chunks.get(tok.i, tok.text)

def svo_triples(doc):
    """Rule-based triples read off the dependency tree.

    For every verb: subjects are its nsubj/nsubjpass children. Objects are its dobj/attr children,
    or the pobj under a prep child ("appeal *against* the acquittal" -> verb "appeal against").
    """
    chunks = {t.i: nc.text for nc in doc.noun_chunks for t in nc}
    triples = []
    for verb in doc:
        if verb.pos_ not in ("VERB", "AUX"):
            continue
        subjects = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")]
        objects = [(verb.lemma_, c) for c in verb.children if c.dep_ in ("dobj", "attr")]
        objects += [(f"{verb.lemma_} {p.text}", o) for p in verb.children if p.dep_ == "prep"
                    for o in p.children if o.dep_ == "pobj"]
        for s in subjects:
            passive = " (passive)" if s.dep_ == "nsubjpass" else ""
            triples += [(phrase(s, chunks), v + passive, phrase(o, chunks)) for v, o in objects]
    return triples

rows = [{"category": c, "sentence": s, "triples": svo_triples(nlp(s))}
        for c, s in zip(picked.category, sentences)]
svo = pd.DataFrame(rows)
print(f"{(svo.triples.str.len() > 0).sum()} of {len(svo)} sentences gave at least one triple, "
      f"{svo.triples.str.len().sum()} triples in total")
svo

# %% [markdown]
# **Reading the table.** Clean hits: *(A strike, cause, traffic chaos)*, *(Michael Howard, back, stem cell
# research)*, *(Security firms, warn about, a Windows virus)*. The failures are instructive:
# - **Row 6, *(The parents, sue, Mart)***: the tokenizer splits "Wal-Mart" at the hyphen, so the noun
#   chunk is just "Mart". *(The parents, sue by, rock group)* is a **PP-attachment error**: "by rock group"
#   belongs to "a CD", but the parser hung it on "suing".
# - **Row 3, *(officials, say on, Tuesday)***: a time adjunct is taken as the object. The real content
#   ("the deficit is set to hit $368bn") sits in a clausal complement (`ccomp`), which our rules skip.
# - **Rows 2 and 19**: the subject is *it*. Knowing that *it* = Virgin Radio needs coreference resolution.
# - **Row 17, *(… Doom, blast, the Golden Joystick awards)***: an appositive ("the ceremony, the Golden
#   Joystick awards") is misread as a second object, and the chunk drops the "3" of "Doom 3".
# - **Rows 9, 12, 13 give nothing**: the action is inside an infinitive or reported-speech clause
#   ("hope *to sign* Juninho", "says … has made him determined"), where the verb has no direct `nsubj`.
#
# Triples are only as good as the parse plus the rules. Real open-information-extraction systems add
# rules for clauses, coordination and coreference.

# %% [markdown]
# ### Two dependency trees

# %%
for s in ["The bank approved the loan for the new factory.",
          "Novelist Andrea Levy won the prize with her book Small Island."]:
    doc = nlp(s)
    print(s, "→", svo_triples(doc))
    displacy.render(doc, style="dep", jupyter=True, options={"compact": True, "distance": 110})

# %%
pd.DataFrame([{"token": t.text, "dep": t.dep_, "head": t.head.text, "POS": t.pos_}
              for t in nlp("The bank approved the loan for the new factory.")])

# %% [markdown]
# Note the attachment choice: "for the new factory" hangs from *approved* or from *loan*. Both
# readings are grammatical; the parser has to pick one.

# %% [markdown]
# ## Part 2 · Semantic search: MiniLM embeddings vs TF-IDF
# Every BBC article (title + body) becomes (a) a MiniLM sentence embedding and (b) a TF-IDF vector from
# our own `src/tfidf.py` on `clean()` text. Both are unit-length, so cosine similarity = dot product.
# MiniLM reads at most 256 word pieces, i.e. roughly the first 200 words; TF-IDF uses the whole article.

# %%
docs = (bbc.title + ". " + bbc.body).tolist()
t = time.perf_counter()
tfidf = TfidfVectorizer(min_df=2)
X_tfidf = tfidf.fit_transform([clean(d) for d in docs])
tfidf_s = time.perf_counter() - t

device = get_device()
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
t = time.perf_counter()
E = st.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
embed_s = time.perf_counter() - t
print(f"TF-IDF {X_tfidf.shape} in {tfidf_s:.1f}s | MiniLM {E.shape} in {embed_s:.1f}s on {device} "
      f"(max_seq_length={st.max_seq_length})")

sim_tfidf = (X_tfidf @ X_tfidf.T).toarray()
sim_minilm = E @ E.T
for S in (sim_tfidf, sim_minilm):
    np.fill_diagonal(S, -1)  # an article is not its own neighbour

# %% [markdown]
# ### 5 query articles, top-3 neighbours each

# %%
queries = bbc.groupby("category").sample(1, random_state=SEED + 1).index
rows = []
for q in queries:
    for method, S in [("TF-IDF", sim_tfidf), ("MiniLM", sim_minilm)]:
        for rank, j in enumerate(np.argsort(-S[q])[:3], 1):
            rows.append({"query": f"[{bbc.category[q]}] {bbc.title[q]}", "method": method, "rank": rank,
                         "neighbour": f"[{bbc.category[j]}] {bbc.title[j]}", "cosine": round(float(S[q, j]), 3)})
pd.DataFrame(rows).set_index(["query", "method", "rank"])

# %% [markdown]
# Cosine values are not comparable across methods: MiniLM scores sit in a different range from
# TF-IDF scores. Only the ranking within one method matters.
#
# ### All 2,089 articles: is the top-3 in the same category?
# A simple automatic check: category precision@3 = share of the 3 nearest articles with the same
# category label.

# %%
cats = bbc.category.to_numpy()
def precision_at_3(S):
    top3 = np.argsort(-S, axis=1)[:, :3]
    return float((cats[top3] == cats[:, None]).mean())

p3 = {"TF-IDF": round(precision_at_3(sim_tfidf), 4), "MiniLM": round(precision_at_3(sim_minilm), 4)}
overlap = np.mean([len(set(np.argsort(-sim_tfidf[i])[:3]) & set(np.argsort(-sim_minilm[i])[:3])) / 3
                   for i in range(len(bbc))])
print("category precision@3:", p3, f"| average top-3 overlap between the methods: {overlap:.1%}")

# %% [markdown]
# On whole articles both methods do well (MiniLM somewhat better): a full article repeats its topic
# words many times, which is exactly what TF-IDF needs. They still agree on fewer than half of the
# neighbours, so they measure "similar" differently. The difference shows up on **short queries that paraphrase**.

# %% [markdown]
# ### Short free-text queries: where meaning beats word overlap

# %%
free_queries = ["carmaker cuts jobs", "handset makers see sales jump",
                "keeping children safe online", "broadband connection speeds",
                "physician shortage in hospitals"]
rows = []
for q in free_queries:
    s_tfidf = (X_tfidf @ tfidf.transform([clean(q)]).T).toarray().ravel()
    s_minilm = E @ st.encode([q], normalize_embeddings=True)[0]
    for method, s in [("TF-IDF", s_tfidf), ("MiniLM", s_minilm)]:
        j = int(np.argmax(s))
        rows.append({"query": q, "method": method, "top result": f"[{bbc.category[j]}] {bbc.title[j]}",
                     "cosine": round(float(s[j]), 3)})
free = pd.DataFrame(rows).set_index(["query", "method"])
free

# %% [markdown]
# **Semantic beats lexical:** for *"carmaker cuts jobs"*, TF-IDF matches the generic word "cuts" and
# returns a politics story about Whitehall job cuts. MiniLM returns *"GM, Ford cut output as sales
# fall"*: it knows GM and Ford are carmakers, although the article need not contain the word "carmaker".
# Likewise *"handset makers"* → TF-IDF finds "sales" in a US retail story; MiniLM finds a mobile-phone
# story. *"keeping children safe online"* → TF-IDF latches onto "children" (asylum children), MiniLM
# onto online safety.
# **Where they agree:** *"broadband connection speeds"* uses the articles' own words, so TF-IDF is
# just as good. **Where neither is convincing:** *"physician shortage in hospitals"*. No article is
# really about that, and both return something loosely related.

# %%
save_results("semantic_search", {
    "svo": {"sentences": len(svo), "with_triple": int((svo.triples.str.len() > 0).sum()),
            "triples": int(svo.triples.str.len().sum()), "examples": svo.to_dict("records")},
    "retrieval": {"n_docs": len(bbc), "category_precision_at_3": p3, "top3_overlap": round(float(overlap), 4),
                  "tfidf_build_s": round(tfidf_s, 1), "minilm_encode_s": round(embed_s, 1), "device": str(device)},
    "free_queries": free.reset_index().to_dict("records"),
})

# %% [markdown]
# ## Viva notes
# **Q1. How do you extract subject–verb–object triples?**
# Parse with spaCy, then for each verb take children labelled `nsubj`/`nsubjpass` as subjects and
# `dobj`/`attr` (or the `pobj` under a `prep`) as objects, and expand each to its noun chunk. It is
# rule-based on top of a neural parser, so parse errors and unusual structures (reported speech,
# coordination) break it.
#
# **Q2. Why can a sentence-transformer find related text with no shared words?**
# It was trained on millions of sentence pairs to put paraphrases close together, so "carmaker" lands
# near "GM" and "Ford". TF-IDF vectors of texts with no word in common have cosine 0 by construction.
#
# **Q3. When is TF-IDF still the better choice?**
# When queries use the documents' exact terms (names, codes, rare keywords). It is also faster, needs no
# GPU and is fully explainable. MiniLM also truncates long documents at 256 word pieces, while TF-IDF
# uses the whole text.
