"""TextRank extractive summarization from scratch.

1. Split the article into sentences.
2. Turn each sentence into a TF-IDF vector (our src/tfidf.py on clean() text).
3. Build a graph: one node per sentence, edge weight = cosine similarity of the two sentences.
4. Run PageRank on the graph: a sentence is important if it is similar to other important sentences.
5. Return the top-k sentences, in their original order so the summary reads naturally.
"""
import nltk
import numpy as np

from src.preprocess import clean
from src.tfidf import TfidfVectorizer


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def pagerank(W: np.ndarray, d: float = 0.85, tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """PageRank by power iteration on a weighted graph.

    W[i, j] = edge weight between nodes i and j. A random surfer at node i moves to neighbour j with
    probability W[i, j] / sum_j W[i, j] (with prob. d), or jumps to a random node (prob. 1 - d).
    A node with no edges ("dangling") jumps to a random node. Same definition as networkx.pagerank.
    """
    n = len(W)
    out = W.sum(axis=1)
    dangling = out == 0
    P = np.divide(W, out[:, None], out=np.zeros_like(W, dtype=float), where=~dangling[:, None])
    r = np.full(n, 1 / n)
    for _ in range(max_iter):
        # rank flowing along edges + rank of dangling nodes spread evenly + random jumps
        r_new = d * (P.T @ r + r[dangling].sum() / n) + (1 - d) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def sentence_graph(sentences: list[str]) -> np.ndarray:
    """Cosine-similarity matrix of the sentences' TF-IDF vectors, with a zero diagonal."""
    cleaned = [clean(s) for s in sentences]
    X = TfidfVectorizer().fit_transform(cleaned)   # rows have length 1 (or are all-zero)
    W = (X @ X.T).toarray()                        # so dot product = cosine similarity
    np.fill_diagonal(W, 0.0)                       # no self-loops
    return W


def textrank_summary(text: str, k: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= k:
        return " ".join(sentences)
    scores = pagerank(sentence_graph(sentences))
    top = sorted(np.argsort(-scores)[:k])          # best k, then back into document order
    return " ".join(sentences[i] for i in top)


def lead_summary(text: str, k: int = 3) -> str:
    """Baseline: the first k sentences (news puts the key facts first)."""
    return " ".join(split_sentences(text)[:k])
