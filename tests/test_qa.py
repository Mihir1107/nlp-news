"""QA metrics, passage chunking and the rule-based router."""
import pandas as pd
import pytest

from src.evaluate import exact_match, normalize_answer, token_f1
from src.qa import chunk_articles


def test_squad_metrics():
    assert normalize_answer("The Vagina Monologues!") == "vagina monologues"
    assert exact_match("the Shrek 2", ["Shrek 2"]) == 1.0
    assert token_f1("Scott Sullivan", ["Mr Sullivan"]) == 0.5
    assert token_f1("Sullivan", ["Scott Sullivan", "Mr Sullivan"]) == pytest.approx(2 / 3)
    assert token_f1("Poland", ["Germany"]) == 0.0


def test_chunks_overlap_and_cover_everything():
    body = " ".join(f"w{i}" for i in range(300))
    df = pd.DataFrame({"id": [7], "category": ["tech"], "title": ["T"], "body": [body]})
    chunks = chunk_articles(df, size=120, stride=80)
    assert [len(c["text"].split()) for c in chunks] == [120, 120, 120, 60]
    assert chunks[0]["text"].split()[80:] == chunks[1]["text"].split()[:40]   # 40-word overlap
    assert chunks[-1]["text"].split()[-1] == "w299"                            # nothing lost
    assert all(c["article_id"] == 7 for c in chunks)


@pytest.fixture(scope="module")
def bot():
    from src.qa import NewsBot
    df = pd.DataFrame({"id": [0, 1], "category": ["sport", "business"],
                       "title": ["Nadal wins", "Oil rises"],
                       "body": ["Rafael Nadal beat Andy Roddick in Seville on Friday. Spain lead 2-0.",
                                "Oil prices rose to $47 a barrel on Thursday. Demand from China grew."]})
    ents = pd.DataFrame({"id": [0, 0], "label": ["PERSON", "PERSON"], "text": ["Rafael Nadal", "Andy Roddick"]})
    try:
        return NewsBot(df, entities=ents, threshold=0.2)
    except OSError:
        pytest.skip("reader model not downloaded")


def test_router_and_reader(bot):
    assert bot.chat("summarize article 1")["intent"] == "summarize"
    assert bot.chat("Who is mentioned in article 0?")["answer"] == "Rafael Nadal, Andy Roddick"
    assert "no article 9" in bot.chat("summarise article 9")["answer"]
    r = bot.chat("Who did Rafael Nadal beat?")
    assert r["intent"] == "question" and r["answer"] == "Andy Roddick" and r["article_id"] == 0
    assert r["passage"][r["answer_start"]:r["answer_end"]] == "Andy Roddick"   # offsets point at the answer
