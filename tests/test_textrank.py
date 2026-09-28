"""src/textrank.py: PageRank must equal networkx; summaries must be well-formed."""
import networkx as nx
import numpy as np

from src.textrank import lead_summary, pagerank, sentence_graph, split_sentences, textrank_summary

TEXT = ("Oil prices rose sharply on Monday. Traders said oil supplies were tight. "
        "The football team won the cup. Oil producers promised to raise supplies of oil. "
        "Analysts expect oil prices to stay high.")


def test_pagerank_matches_networkx():
    rng = np.random.default_rng(0)
    W = rng.random((6, 6)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    W[5, :] = W[:, 5] = 0                      # one isolated (dangling) node
    ours = pagerank(W)
    ref = nx.pagerank(nx.from_numpy_array(W), alpha=0.85, tol=1e-12, max_iter=1000)
    assert np.allclose(ours, [ref[i] for i in range(6)], atol=1e-6)
    assert np.isclose(ours.sum(), 1.0)


def test_graph_is_symmetric_with_zero_diagonal():
    W = sentence_graph(split_sentences(TEXT))
    assert np.allclose(W, W.T) and not np.diag(W).any()
    assert W[2].sum() == 0                     # the football sentence shares no words


def test_summary_keeps_order_and_picks_central_sentences():
    summary = textrank_summary(TEXT, k=2)
    assert "football" not in summary           # off-topic sentence is least central
    sents = split_sentences(TEXT)
    picked = [s for s in sents if s in summary]
    assert len(picked) == 2 and sents.index(picked[0]) < sents.index(picked[1])


def test_short_text_and_lead():
    assert textrank_summary("One sentence only.", k=3) == "One sentence only."
    assert lead_summary(TEXT, k=1) == "Oil prices rose sharply on Monday."


def test_rank_scores_match_summary():
    from src.textrank import textrank_rank
    sents, scores = textrank_rank(TEXT)
    assert len(sents) == len(scores) == 5 and np.isclose(scores.sum(), 1.0)
    best = sents[int(np.argmax(scores))]
    assert best in textrank_summary(TEXT, k=1)
