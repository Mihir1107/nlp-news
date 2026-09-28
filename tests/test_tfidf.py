"""src/tfidf.py must reproduce sklearn exactly on a toy corpus."""
import math

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf

from src.tfidf import BagOfWords, TfidfVectorizer

TOY = [
    "stocks fall as oil prices rise",
    "oil prices rise again",
    "team wins the cup final",
    "stocks rise after team sponsor deal",
    "the final deal on oil",
]
SK = dict(tokenizer=str.split, lowercase=False, token_pattern=None)  # same tokens as ours


def test_bow_matches_sklearn():
    for ngr in [(1, 1), (1, 2)]:
        ours, sk = BagOfWords(ngram_range=ngr), CountVectorizer(ngram_range=ngr, **SK)
        assert np.allclose(ours.fit_transform(TOY).toarray(), sk.fit_transform(TOY).toarray())
        assert list(ours.get_feature_names_out()) == list(sk.get_feature_names_out())


def test_tfidf_matches_sklearn():
    for ngr in [(1, 1), (1, 2)]:
        ours, sk = TfidfVectorizer(ngram_range=ngr), SkTfidf(ngram_range=ngr, **SK)
        assert np.allclose(ours.fit_transform(TOY).toarray(), sk.fit_transform(TOY).toarray())
        assert np.allclose(ours.idf_, sk.idf_)


def test_min_df_matches_sklearn():
    ours, sk = TfidfVectorizer(min_df=2), SkTfidf(min_df=2, **SK)
    assert np.allclose(ours.fit_transform(TOY).toarray(), sk.fit_transform(TOY).toarray())


def test_idf_by_hand():
    v = TfidfVectorizer().fit(TOY)
    # "oil" is in 3 of 5 documents: ln(6/4) + 1
    assert math.isclose(v.idf_[v.vocabulary_["oil"]], math.log(6 / 4) + 1)
    # "cup" is in 1 document: ln(6/2) + 1
    assert math.isclose(v.idf_[v.vocabulary_["cup"]], math.log(6 / 2) + 1)


def test_rows_unit_length_and_unknown_words_ignored():
    v = TfidfVectorizer().fit(TOY)
    X = v.transform(["oil stocks zebra", "zebra"]).toarray()
    assert math.isclose(np.linalg.norm(X[0]), 1.0)
    assert not X[1].any()  # only unknown words -> all-zero row, no crash
