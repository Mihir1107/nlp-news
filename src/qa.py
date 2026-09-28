"""NewsBot: question answering over BBC articles, plus a RULE-BASED intent router.

Pipeline for a factual question (retrieve, then read):
  1. Retriever: TF-IDF cosine (our src/tfidf.py) finds the top-k ~120-word passages.
  2. Reader: DistilBERT fine-tuned on SQuAD scores every token as a possible answer start and end.
     The best span across all k passages (highest start_logit + end_logit, at most 30 tokens) wins.
  3. If the reader's confidence is below a threshold, NewsBot says "I don't know".
The router in NewsBot.chat() is plain regular expressions, not a learned model.
"""
import re

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.preprocess import clean
from src.textrank import textrank_summary
from src.tfidf import TfidfVectorizer

READER = "distilbert/distilbert-base-cased-distilled-squad"


def chunk_articles(df, size: int = 120, stride: int = 80) -> list[dict]:
    """Split each article body into ~size-word passages; consecutive passages overlap by size - stride
    words, so an answer that crosses a boundary is still complete in one passage."""
    passages = []
    for row in df.itertuples():
        words, start = row.body.split(), 0
        while True:
            passages.append({"article_id": row.id, "category": row.category, "title": row.title,
                             "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += stride
    return passages


class Retriever:
    """TF-IDF over title + passage. clean(lemma=False) keeps start-up fast (no POS tagging)."""

    def __init__(self, passages: list[dict]):
        self.passages = passages
        self.vec = TfidfVectorizer(min_df=1)
        self.X = self.vec.fit_transform([clean(f"{p['title']}. {p['text']}", lemma=False) for p in passages])

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scores = (self.X @ self.vec.transform([clean(query, lemma=False)]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class Reader:
    def __init__(self, device="cpu", max_answer_tokens: int = 30):
        self.tok = AutoTokenizer.from_pretrained(READER)
        self.model = AutoModelForQuestionAnswering.from_pretrained(READER).to(device).eval()
        self.device, self.max_len = device, max_answer_tokens

    @torch.no_grad()
    def best_span(self, question: str, contexts: list[str]) -> dict:
        enc = self.tok([question] * len(contexts), contexts, truncation="only_second", max_length=384,
                       padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        out = self.model(**enc.to(self.device))
        best = {"score": -np.inf}
        for p in range(len(contexts)):
            ctx = [i for i, s in enumerate(enc.sequence_ids(p)) if s == 1]  # passage tokens only
            s, e = out.start_logits[p, ctx].cpu(), out.end_logits[p, ctx].cpu()
            pair = s[:, None] + e[None, :]                     # score of span (start i, end j)
            n = len(ctx)
            valid = torch.ones(n, n).triu() - torch.ones(n, n).triu(self.max_len)  # i <= j < i+30
            pair[valid == 0] = -torch.inf
            i, j = divmod(int(pair.argmax()), n)
            if pair[i, j] > best["score"]:
                prob = float(s.softmax(0)[i] * e.softmax(0)[j])  # P(start) * P(end) in this passage
                a, b = offsets[p, ctx[i], 0], offsets[p, ctx[j], 1]
                best = {"score": float(pair[i, j]), "confidence": prob, "passage": p,
                        "answer": contexts[p][a:b]}
        return best


class NewsBot:
    def __init__(self, articles, entities=None, classifier=None, device="cpu", threshold=0.2, k=3):
        self.articles = articles.set_index("id")
        self.entities, self.classifier = entities, classifier
        self.retriever = Retriever(chunk_articles(articles))
        self.reader = Reader(device)
        self.threshold, self.k = threshold, k

    def ask(self, question: str) -> dict:
        hits = self.retriever.search(question, self.k)
        passages = [self.retriever.passages[i] for i, _ in hits]
        span = self.reader.best_span(question, [p["text"] for p in passages])
        src = passages[span["passage"]]
        known = span["confidence"] >= self.threshold
        return {"intent": "question", "answer": span["answer"] if known else "I don't know.",
                "best_guess": span["answer"], "confidence": round(span["confidence"], 4),
                "article_id": int(src["article_id"]), "category": src["category"], "title": src["title"],
                "retrieved_article_ids": [int(p["article_id"]) for p in passages]}

    # ---- Rule-based intent router (regular expressions, no learning) ----
    def chat(self, message: str) -> dict:
        m = re.search(r"summari[sz]e article (\d+)", message, re.I)
        if m:
            return self._with_article(int(m[1]), "summarize",
                                      lambda a: textrank_summary(a.body, k=3))
        m = re.search(r"who (?:is|are) mentioned in article (\d+)", message, re.I)
        if m and self.entities is not None:
            def people(a):
                ents = self.entities[(self.entities.id == a.name) & (self.entities.label == "PERSON")]
                names = ents.text.str.replace(r"['’]s$", "", regex=True).value_counts()
                return ", ".join(names.index[:10]) or "No people found."
            return self._with_article(int(m[1]), "entities", people)
        m = re.search(r"what (?:topic|category) is (?:article (\d+)|this)\W*(.*)", message, re.I | re.S)
        if m and self.classifier is not None:
            if m[1]:
                return self._with_article(int(m[1]), "topic", lambda a: self._topic(f"{a.title}. {a.body}")
                                          + f" (BBC label: {a.category})")
            return {"intent": "topic", "answer": self._topic(m[2]) if m[2].strip() else "Paste the text after 'this:'."}
        return self.ask(message)

    def _topic(self, text: str) -> str:
        probs = self.classifier.predict_proba([clean(text)])[0]
        label = self.classifier.classes_[probs.argmax()]
        names = ["World", "Sports", "Business", "Sci/Tech"]  # AG News classes of the TF-IDF model
        return f"{names[label]} ({probs.max():.0%})"

    def _with_article(self, article_id: int, intent: str, fn) -> dict:
        if article_id not in self.articles.index:
            return {"intent": intent, "answer": f"There is no article {article_id} "
                                                f"(ids 0-{self.articles.index.max()})."}
        a = self.articles.loc[article_id]
        return {"intent": intent, "article_id": article_id, "title": a.title, "answer": fn(a)}
