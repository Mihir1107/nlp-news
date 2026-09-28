"""src/ngram.py checked against probabilities worked out by hand on a toy corpus."""
import math

import pytest

from src.ngram import BOS, EOS, UNK, NGramLM, sentences_from_text

TOY = [["i", "like", "tea"], ["i", "like", "coffee"], ["you", "like", "tea"]]


def test_bigram_mle_by_hand():
    lm = NGramLM(n=2, min_count=1).fit(TOY)
    assert lm.prob("i", (BOS,)) == 2 / 3        # 2 of 3 sentences start with "i"
    assert lm.prob("like", ("i",)) == 1.0       # "i" is always followed by "like"
    assert lm.prob("tea", ("like",)) == 2 / 3
    assert lm.prob(EOS, ("tea",)) == 1.0
    # P(i like tea) = 2/3 * 1 * 2/3 * 1 = 4/9
    assert math.isclose(lm.logprob(["i", "like", "tea"]), math.log(4 / 9))


def test_unigram_by_hand():
    lm = NGramLM(n=1, min_count=1).fit(TOY)
    assert lm.prob("like", ()) == 3 / 12        # 9 words + 3 </s> = 12 tokens


def test_laplace_by_hand():
    lm = NGramLM(n=2, smoothing="laplace", min_count=1).fit(TOY)
    V = 7                                        # i like tea coffee you <UNK> </s>
    assert len(lm.vocab) == V
    assert math.isclose(lm.prob("like", ("i",)), (2 + 1) / (2 + V))
    assert math.isclose(lm.prob("coffee", ("i",)), (0 + 1) / (2 + V))


def test_zero_probability_problem():
    mle = NGramLM(n=2, min_count=1).fit(TOY)
    lap = NGramLM(n=2, smoothing="laplace", min_count=1).fit(TOY)
    s = ["you", "like", "coffee", "tea"]         # "coffee tea" never seen
    assert mle.logprob(s) == float("-inf")
    assert math.isfinite(lap.logprob(s))


def test_unk_mapping():
    lm = NGramLM(n=1, min_count=2).fit(TOY)     # coffee, you seen once -> <UNK>
    assert "coffee" not in lm.vocab and UNK in lm.vocab
    assert lm.prob(UNK, ()) == 2 / 12
    assert lm.logprob(["zebra"]) == lm.logprob(["coffee"])  # both are <UNK>


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("smoothing", ["none", "laplace", "addk"])
def test_distribution_sums_to_one(n, smoothing):
    lm = NGramLM(n=n, smoothing=smoothing, k=0.1, min_count=1).fit(TOY)
    for history in list(lm.counts)[:5]:         # histories seen in training
        assert math.isclose(sum(lm.distribution(history).values()), 1.0)


def test_perplexity_and_generate():
    lm = NGramLM(n=2, min_count=1).fit(TOY)
    # training perplexity: logprobs of the 3 sentences over 12 predicted tokens
    expected = math.exp(-(math.log(4 / 9) + math.log(2 / 9) + math.log(1 / 3 * 2 / 3)) / 12)
    assert math.isclose(lm.perplexity(TOY), expected)
    words = lm.generate(seed=0).split()
    assert words and all(w in lm.vocab for w in words)


def test_sentences_from_text():
    assert sentences_from_text("Mr. Brown won. Shares fell 3%!") == [["mr", "brown", "won"], ["shares", "fell", "3"]]
