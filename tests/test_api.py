"""API tests with FastAPI's TestClient (runs the real start-up, so models must be available)."""
import pytest
from fastapi.testclient import TestClient

from src.utils import MODELS

pytestmark = pytest.mark.skipif(not (MODELS / "distilbert-agnews").exists(),
                                reason="trained models missing: run notebooks 02, 04, 07")

SPORT = "Manchester United beat Chelsea 2-1 at Old Trafford, with Rooney scoring the winner in the final minute."


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:  # "with" runs the lifespan start-up and shutdown
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "bart" not in r.json()["models_loaded"]          # DistilBART is lazy


@pytest.mark.parametrize("model", ["tfidf", "distilbert"])
def test_classify(client, model):
    r = client.post("/classify", json={"text": SPORT, "model": model})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == "Sports" and body["model"] == model
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["latency_ms"] > 0 and body["truncated"] is False


def test_classify_default_is_tfidf(client):
    assert client.post("/classify", json={"text": SPORT}).json()["model"] == "tfidf"


def test_long_input_is_truncated_for_distilbert(client):
    r = client.post("/classify", json={"text": "The market rallied again today. " * 150, "model": "distilbert"})
    assert r.status_code == 200 and r.json()["truncated"] is True


@pytest.mark.parametrize("payload", [
    {"text": ""},                          # too short
    {"text": "   "},                       # only whitespace
    {"text": "x" * 10_001},                # too long
    {"text": "ok", "model": "gpt5"},       # unknown model
    {},                                    # missing field
])
def test_validation_422(client, payload):
    assert client.post("/classify", json=payload).status_code == 422


def test_summarize_textrank(client):
    text = ("Oil prices rose sharply on Monday. Traders said supplies were tight. The football team won. "
            "Oil producers promised to raise supplies. Analysts expect oil prices to stay high.")
    r = client.post("/summarize", json={"text": text, "sentences": 2})
    body = r.json()
    assert r.status_code == 200 and "football" not in body["summary"]
    assert len(body["sentences"]) == len(body["scores"]) == 5 and len(body["selected"]) == 2
    assert all(body["sentences"][i] in body["summary"] for i in body["selected"])


def test_summarize_distilbart_lazy_load(client):
    text = ("The Bank of England kept interest rates at 4.75% on Thursday, saying inflation remained "
            "under control. Economists had expected no change. House prices have cooled in recent months, "
            "and retail sales slowed over Christmas. The bank will publish its inflation report next week.")
    r = client.post("/summarize", json={"text": text, "method": "distilbart"})
    assert r.status_code == 200 and len(r.json()["summary"].split()) > 5
    assert "bart" in client.get("/health").json()["models_loaded"]


def test_entities(client):
    ents = client.post("/entities", json={"text": "Tony Blair met Jacques Chirac in Paris."}).json()["entities"]
    assert {"text": "Tony Blair", "label": "PERSON"} in [{k: e[k] for k in ("text", "label")} for e in ents]
    assert any(e["label"] == "GPE" and e["text"] == "Paris" for e in ents)


def test_ask(client):
    r = client.post("/ask", json={"question": "Who is the head of Tiga?"}).json()
    assert r["intent"] == "question" and r["answer"] == "Fred Hasson"
    assert r["passage"][r["answer_start"]:r["answer_end"]] == "Fred Hasson"
    r = client.post("/ask", json={"question": "summarize article 5"}).json()
    assert r["intent"] == "summarize" and r["article_id"] == 5
    assert client.post("/ask", json={"question": " "}).status_code == 422


def test_results_and_samples(client):
    res = client.get("/results").json()
    assert {r["key"] for r in res["classification"]} == {"bow", "tfidf", "word2vec", "glove", "bilstm", "distilbert"}
    assert 0 < res["qa"]["reader_f1"] <= 1 and len(res["summarization"]) == 3
    s = client.get("/samples").json()
    assert len(s["headlines"]) == 4 and len(s["articles"]) == 3 and s["questions"]


def test_web_app_served_without_shadowing_api(client):
    from api.app import FRONTEND
    if not FRONTEND.exists():
        pytest.skip("web app not built: cd frontend && npm install && npm run build")
    page = client.get("/")
    assert page.status_code == 200 and "<div id=\"root\">" in page.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/health").json()["status"] == "ok"       # API routes still win over the static mount
