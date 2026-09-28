"""Unit tests for src/preprocess.py on hand-written examples."""
import pytest

from src.preprocess import (clean, lemmatize, regex_tokenize, remove_punctuation,
                            remove_stopwords, replace_emoji, shorten_elongated, split_hashtags,
                            tokenize)


def test_tokenizers_differ_on_contractions():
    assert tokenize("don't") == ["do", "n't"]
    assert regex_tokenize("don't") == ["don't"]


def test_remove_punctuation_keeps_words():
    assert remove_punctuation(["hello", ",", "--", "n't", "u.s.", "“"]) == ["hello", "n't", "u.s."]


def test_stopwords_keep_negation():
    assert remove_stopwords(["the", "movie", "was", "not", "good"]) == ["movie", "not", "good"]


def test_lemmatize_uses_pos():
    assert lemmatize(["he", "was", "running"]) == ["he", "be", "run"]
    assert lemmatize(["the", "wolves", "ran"]) == ["the", "wolf", "run"]
    # The POS tag decides the lemma: alone, "better" is tagged adverb (RBR) -> "well".
    assert lemmatize(["better"]) == ["well"]


def test_lemmatize_short_word_guard():
    # plain WordNet gives "u", "ha", "bos"; the guard keeps the original word
    assert lemmatize(["us"]) == ["us"]
    assert lemmatize(["boss"]) == ["boss"]
    assert lemmatize(["cars"]) == ["car"]           # normal plurals still work


def test_clean_news():
    out = clean("The markets were NOT rising, analysts said.")
    assert out == "market not rise analyst say"


def test_possessive_removed():
    assert clean("Nestle's profit") == "nestle profit"


def test_clean_keeps_numbers_and_is_idempotent_on_empty():
    assert clean("") == ""
    assert "2004" in clean("Profits rose in 2004.")


def test_tweet_steps():
    assert split_hashtags("#HappyBirthdayRemus") == "Happy Birthday Remus"
    assert shorten_elongated("soooo gooood") == "soo good"
    assert replace_emoji("great 😂").split() == ["great", "laugh"]
    assert replace_emoji("ok :)").split() == ["ok", "happy"]
    assert replace_emoji("x 🦄 y").split() == ["x", "y"]  # unknown emoji removed


def test_clean_tweet():
    out = clean("@user Soooo happy!!! #GameDay 😍 http://t.co/xyz", mode="tweet")
    assert out.split() == ["user", "soo", "happy", "game", "day", "love", "url"]


def test_tweet_ablation_switches():
    t = "#GameDay 😍"
    assert clean(t, mode="tweet") == "game day love"
    assert clean(t, mode="tweet", handle_emoji=False, handle_hashtags=False) == "gameday"


def test_bad_mode():
    with pytest.raises(ValueError):
        clean("x", mode="blog")
