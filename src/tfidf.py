"""Bag-of-Words and TF-IDF from scratch, built to give exactly sklearn's numbers.

Input documents are already-cleaned strings; a term is a whitespace-separated word
(or a run of n words when ngram_range allows n-grams).

BoW:     X[d, t] = count of term t in document d
TF-IDF:  X[d, t] = count(t, d) * idf(t), then every row is scaled to length 1 (L2 norm)
         idf(t)  = ln((1 + n) / (1 + df(t))) + 1          (sklearn's default, smooth_idf=True)
         n = number of documents, df(t) = number of documents that contain t.
  - The "+1" inside the log acts as if one extra document contained every term,
    so a term that appears in every document never divides by zero.
  - The "+1" outside keeps idf > 0, so such a term still counts a little.
Because rows have length 1, the cosine similarity of two documents is just their dot product.
"""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, diags


class BagOfWords:
    def __init__(self, ngram_range=(1, 1), min_df=1):
        self.ngram_range = ngram_range  # (1, 2) = unigrams and bigrams
        self.min_df = min_df            # ignore terms found in fewer documents than this

    def _terms(self, doc: str) -> list[str]:
        """All n-grams of the document: 'a b c' with (1, 2) -> a, b, c, 'a b', 'b c'."""
        words = doc.split()
        lo, hi = self.ngram_range
        return [" ".join(words[i:i + n]) for n in range(lo, hi + 1) for i in range(len(words) - n + 1)]

    def fit(self, docs):
        df = Counter()
        for doc in docs:
            df.update(set(self._terms(doc)))  # set(): count each term once per document
        terms = sorted(t for t, c in df.items() if c >= self.min_df)  # sorted, like sklearn
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        self.df_ = np.array([df[t] for t in terms])
        self.n_docs_ = len(docs)
        return self

    def transform(self, docs) -> csr_matrix:
        """Count matrix, shape (n_docs, vocab_size). Terms not in the vocabulary are ignored."""
        rows, cols, vals = [], [], []
        for d, doc in enumerate(docs):
            counts = Counter(t for t in self._terms(doc) if t in self.vocabulary_)
            for term, c in counts.items():
                rows.append(d); cols.append(self.vocabulary_[term]); vals.append(c)
        return csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocabulary_)), dtype=float)

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def get_feature_names_out(self):
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.get))


class TfidfVectorizer(BagOfWords):
    def fit(self, docs):
        super().fit(docs)
        self.idf_ = np.log((1 + self.n_docs_) / (1 + self.df_)) + 1
        return self

    def transform(self, docs) -> csr_matrix:
        X = super().transform(docs)          # raw term counts (tf)
        X = X @ diags(self.idf_)             # multiply column t by idf(t)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0              # empty document: leave the zero row alone
        return csr_matrix(diags(1 / norms) @ X)  # divide each row by its length
