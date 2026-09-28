"""NewsMind REST API (FastAPI).

Run from the repo root:   uvicorn api.app:app
Interactive docs (Swagger UI): http://127.0.0.1:8000/docs

Endpoints
  GET  /health      which models are loaded
  POST /classify    news topic (AG News classes) with TF-IDF+LR (default) or fine-tuned DistilBERT
  POST /summarize   TextRank (default, fast) or DistilBART (loaded on first use, ~1.2 GB)
  POST /entities    named entities with spaCy
  POST /ask         NewsBot: rule-based router + TF-IDF retrieval + DistilBERT-SQuAD reader
"""
import json
import time
from contextlib import asynccontextmanager
from typing import Literal

import src.utils  # noqa: F401  (FIRST: cache folders)
import joblib
import pandas as pd
import spacy
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from src.data import AG_LABELS, load
from src.preprocess import clean
from src.qa import NewsBot
from src.textrank import textrank_summary
from src.utils import DATA, MODELS, RESULTS, get_device

MAX_CHARS = 10_000
BART_NAME = "sshleifer/distilbart-cnn-12-6"
state: dict = {}  # models loaded once at start-up, shared by all requests


# ---- Request / response schemas (FastAPI returns 422 when a request breaks them) ----
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS, description="1 to 10,000 characters")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


class ClassifyIn(TextIn):
    model: Literal["tfidf", "distilbert"] = "tfidf"


class ClassifyOut(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    truncated: bool = Field(description="True if the text was cut to the model's maximum input length")
    latency_ms: float


class SummarizeIn(TextIn):
    method: Literal["textrank", "distilbart"] = "textrank"
    sentences: int = Field(3, ge=1, le=10, description="TextRank only: sentences to keep")


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1_000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must contain non-whitespace characters")
        return v


# ---- Start-up: load models once ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in [MODELS / "tfidf_lr.joblib", MODELS / "distilbert-agnews", DATA / "entities.parquet"]:
        if not path.exists():
            raise RuntimeError(f"{path} missing: run notebooks 02, 04 and 07 first")
    device = get_device()
    state["device"] = device
    state["tfidf"] = joblib.load(MODELS / "tfidf_lr.joblib")
    state["bert_tok"] = AutoTokenizer.from_pretrained(MODELS / "distilbert-agnews")
    state["bert"] = AutoModelForSequenceClassification.from_pretrained(MODELS / "distilbert-agnews").to(device).eval()
    state["nlp"] = spacy.load("en_core_web_sm")
    threshold = json.loads((RESULTS / "qa.json").read_text())["threshold"]  # tuned in nb10
    state["bot"] = NewsBot(load("bbc"), entities=pd.read_parquet(DATA / "entities.parquet"),
                           classifier=state["tfidf"], device=device, threshold=threshold)
    # Warm-up: NLTK loads its tagger/WordNet and PyTorch compiles GPU kernels on first use.
    # Doing it here keeps the first real request fast.
    classify(ClassifyIn(text="Warm-up sentence about markets.", model="tfidf"))
    classify(ClassifyIn(text="Warm-up sentence about markets.", model="distilbert"))
    yield
    state.clear()


app = FastAPI(title="NewsMind API", version="1.0", lifespan=lifespan,
              description="News classification, summarization, NER and question answering.")


def ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(state["device"]),
            "models_loaded": [k for k in ("tfidf", "bert", "nlp", "bot", "bart") if k in state]}


@app.post("/classify", response_model=ClassifyOut)
def classify(req: ClassifyIn):
    start = time.perf_counter()
    if req.model == "tfidf":
        probs = state["tfidf"].predict_proba([clean(req.text)])[0]  # same cleaning as in training
        truncated = False
    else:
        tok = state["bert_tok"]
        n_tokens = len(tok(req.text).input_ids)
        # DistilBERT reads at most 512 tokens: anything after that is cut off (and we say so).
        enc = tok(req.text, truncation=True, max_length=512, return_tensors="pt").to(state["device"])
        with torch.no_grad():
            probs = state["bert"](**enc).logits.softmax(-1)[0].cpu().numpy()
        truncated = n_tokens > 512
    best = int(probs.argmax())
    return ClassifyOut(label=AG_LABELS[best], confidence=round(float(probs[best]), 4),
                       probabilities={l: round(float(p), 4) for l, p in zip(AG_LABELS, probs)},
                       model=req.model, truncated=truncated, latency_ms=ms_since(start))


@app.post("/summarize")
def summarize(req: SummarizeIn):
    start = time.perf_counter()
    if req.method == "textrank":
        return {"summary": textrank_summary(req.text, k=req.sentences), "method": "textrank",
                "truncated": False, "latency_ms": ms_since(start)}
    if "bart" not in state:  # lazy load: only the first DistilBART request pays for it
        state["bart_tok"] = AutoTokenizer.from_pretrained(BART_NAME)
        state["bart"] = AutoModelForSeq2SeqLM.from_pretrained(BART_NAME).to(state["device"]).eval()
    tok = state["bart_tok"]
    n_tokens = len(tok(req.text).input_ids)
    enc = tok(req.text, truncation=True, max_length=1024, return_tensors="pt").to(state["device"])
    with torch.no_grad():
        out = state["bart"].generate(**enc, num_beams=4, min_length=min(40, n_tokens // 2), max_length=130,
                                     no_repeat_ngram_size=3, early_stopping=True)
    return {"summary": tok.decode(out[0], skip_special_tokens=True).strip(), "method": "distilbart",
            "truncated": n_tokens > 1024, "latency_ms": ms_since(start)}


@app.post("/entities")
def entities(req: TextIn):
    start = time.perf_counter()
    doc = state["nlp"](req.text)
    return {"entities": [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                         for e in doc.ents], "latency_ms": ms_since(start)}


@app.post("/ask")
def ask(req: AskIn):
    start = time.perf_counter()
    return {**state["bot"].chat(req.question), "latency_ms": ms_since(start)}
