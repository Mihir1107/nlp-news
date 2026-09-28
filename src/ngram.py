"""N-gram language model from scratch.

A sentence w1..wm is padded as  <s> ... <s> w1 .. wm </s>  (n-1 start markers) and
    P(sentence) = product over i of P(w_i | the n-1 words before it).
The n-1 previous words are the "history" h.

    MLE ("none"):  P(w | h) = c(h, w) / c(h)
    add-k:         P(w | h) = (c(h, w) + k) / (c(h) + k * V)     "laplace" is add-k with k = 1

c(h, w) = how often w followed h in training, c(h) = how often h was followed by anything,
V = vocabulary size: known words + <UNK> + </s>. (<s> is never predicted, so it is not in V.)
Training words seen fewer than min_count times become <UNK>. An unseen word at test time
also becomes <UNK>, so it gets a probability instead of breaking the model.
"""
import math
import random
import re
from collections import Counter, defaultdict

import nltk

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


def sentences_from_text(text: str) -> list[list[str]]:
    """Split text into sentences, each a list of lowercase word tokens (punctuation dropped)."""
    return [toks for s in nltk.sent_tokenize(text)
            if (toks := re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower()))]


class NGramLM:
    def __init__(self, n: int = 2, smoothing: str = "none", k: float = 1.0, min_count: int = 2):
        if smoothing not in ("none", "laplace", "addk"):
            raise ValueError("smoothing must be 'none', 'laplace' or 'addk'")
        self.n = n
        self.smoothing = smoothing
        self.k = 1.0 if smoothing == "laplace" else k
        self.min_count = min_count

    def _pad(self, sentence: list[str]) -> list[str]:
        """Map unknown words to <UNK> and add the start/end markers."""
        words = [w if w in self.vocab else UNK for w in sentence]
        return [BOS] * (self.n - 1) + words + [EOS]

    def _history(self, tokens: list[str], i: int) -> tuple:
        """The n-1 tokens before position i (empty tuple for a unigram model)."""
        return tuple(tokens[i - self.n + 1:i])

    def fit(self, sentences: list[list[str]]):
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= self.min_count} | {UNK, EOS}
        self.counts = defaultdict(Counter)  # counts[history][word] = c(history, word)
        for s in sentences:
            tokens = self._pad(s)
            for i in range(self.n - 1, len(tokens)):
                self.counts[self._history(tokens, i)][tokens[i]] += 1
        self.totals = {h: sum(c.values()) for h, c in self.counts.items()}  # c(history)
        return self

    def prob(self, word: str, history: tuple) -> float:
        """P(word | history). `history` must be the last n-1 tokens."""
        c = self.counts[history][word] if history in self.counts else 0
        total = self.totals.get(history, 0)
        if self.smoothing == "none":
            return c / total if total else 0.0  # unseen history: no information -> 0
        return (c + self.k) / (total + self.k * len(self.vocab))

    def distribution(self, history: tuple) -> dict[str, float]:
        """P(w | history) for every w in the vocabulary (sums to 1 when defined)."""
        return {w: self.prob(w, history) for w in self.vocab}

    def logprob(self, sentence: list[str]) -> float:
        """Natural-log probability of the sentence, including </s>. -inf if any factor is 0."""
        tokens, lp = self._pad(sentence), 0.0
        for i in range(self.n - 1, len(tokens)):
            p = self.prob(tokens[i], self._history(tokens, i))
            if p == 0.0:
                return float("-inf")  # the zero-probability problem
            lp += math.log(p)
        return lp

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(-average log-prob per predicted token). Lower = less surprised. inf if any P = 0."""
        total_lp = sum(self.logprob(s) for s in sentences)
        n_tokens = sum(len(s) + 1 for s in sentences)  # +1 for </s>
        return math.exp(-total_lp / n_tokens)

    def generate(self, max_words: int = 25, seed: int | None = None) -> str:
        """Sample a sentence word by word from the *observed* counts after the current history.

        We sample from the counts (not the smoothed distribution) because add-k gives every
        vocabulary word a little mass, and sampling from it mostly produces random rare words.
        """
        rng = random.Random(seed)
        tokens, out = [BOS] * (self.n - 1), []
        while len(out) < max_words:
            options = {w: c for w, c in self.counts[self._history(tokens, len(tokens))].items()
                       if w != UNK}
            if not options:  # only <UNK> ever followed this history
                break
            word = rng.choices(list(options), weights=list(options.values()))[0]
            if word == EOS:
                break
            out.append(word)
            tokens.append(word)
        return " ".join(out)
