"""Datasets: repair known raw-text artefacts and build the fixed splits.

Every notebook loads data through `load(name)`, so all models see identical rows.
Raw parquet files in data/raw/ are written by scripts/download_data.py.

Split files (data/splits/):
  ag_train       30,000 AG News train rows (stratified, 7,500 per class)
  ag_train_20k   20,000-row stratified subset of ag_train (DistilBERT fine-tuning)
  ag_val          2,000 AG rows NOT in ag_train (for watching LSTM/BERT training)
  ag_test         7,600 AG test rows (the full official test set)
  bbc             2,089 BBC articles (deduplicated) + reference summaries
  tweet_train / tweet_val / tweet_test   TweetEval sentiment, official splits
"""
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import FAST, RAW, SEED, SPLITS

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
BBC_LABELS = ["business", "entertainment", "politics", "sport", "tech"]
TWEET_LABELS = ["negative", "neutral", "positive"]

FAST_PER_CLASS = 500  # rows per class kept when NEWSMIND_FAST=1


# ---- AG News text repair ---------------------------------------------------
def repair_ag_text(text: str) -> str:
    """Undo AG News scraping artefacts. This is data repair, not NLP preprocessing."""
    text = text.replace("\\", " ")                     # escaped newlines: "dwindling\band"
    text = text.replace(" #39;", "'")                  # "the Web #39;s" -> "the Web's"
    text = re.sub(r"#(\d+);", lambda m: chr(int(m[1])), text)  # "#36;10" -> "$10"
    text = re.sub(r"(\w) amp;(\w)", r"\1&\2", text)    # "AT amp;T" -> "AT&T"
    text = text.replace("amp;", "&").replace("quot;", '"')
    text = re.sub(r"(?<![&\w])(hellip|nbsp|rsquo|lsquo|mdash|ndash|lt|gt);",  # bare "hellip;"
                  lambda m: html.unescape(f"&{m[1]};"), text)
    text = html.unescape(text)                         # "&lt;b&gt;" -> "<b>"
    text = re.sub(r"<[^>]+>", " ", text)               # drop HTML tags
    return re.sub(r"\s+", " ", text).strip()


# ---- BBC text repair -------------------------------------------------------
# The HF version of the BBC dataset turned every newline into ".", so the title
# is joined to the body by ".." and paragraphs are glued: "funds...Duncan".
# Summaries also glue sentences with no space: "£50,000.Mr Bannatyne".
_GLUED = re.compile(r"(?<=[a-z0-9%)\"'])([.!?])(?=[A-Z\"])")  # "000.Mr" but not "U.S"


def fix_glued_sentences(text: str) -> str:
    """Insert the missing space after a sentence end: 'funds.Duncan' -> 'funds. Duncan'."""
    return _GLUED.sub(r"\1 ", text)


def parse_bbc_article(raw: str) -> tuple[str, str]:
    """Split a raw BBC article into (title, body) and restore paragraph breaks."""
    raw = html.unescape(raw)                                   # 22 articles: "Jos&#233;" -> "José"
    title, _, body = raw.partition("..")                       # first ".." ends the title
    body = re.sub(r"(?<=[.!?\"'])\.\.(?=\S)", " ", body)       # "funds...Duncan"
    body = re.sub(r"\.\.(?=[A-Z\"'])", ". ", body)             # un-punctuated subheadings
    return title.strip(), fix_glued_sentences(body).strip()


# ---- TweetEval text repair -------------------------------------------------
def repair_tweet_text(text: str) -> str:
    """Decode literal escape text like '\\u2019' or '\\ud83d\\ude02' into real characters.

    11% of train tweets contain these escapes. Emoji are stored as two escaped
    UTF-16 halves (a surrogate pair), so we rebuild the string through UTF-16.
    """
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), text)
    return text.encode("utf-16", "surrogatepass").decode("utf-16")


# ---- Build splits (run once by scripts/download_data.py) -------------------
def _stratified_take(df, n, label="label"):
    """Return (n stratified rows, the remaining rows). Deterministic via SEED."""
    return train_test_split(df, train_size=n, stratify=df[label], random_state=SEED)


def build_splits() -> dict[str, pd.DataFrame]:
    """Create every split from data/raw/ and save to data/splits/ as parquet."""
    out = {}

    # AG News: keep the original row index so every row is traceable.
    ag = {}
    for part in ["train", "test"]:
        df = pd.read_parquet(RAW / f"ag_{part}.parquet")
        df = df.rename(columns={"text": "text_raw"}).rename_axis("orig_idx").reset_index()
        df["text"] = df["text_raw"].map(repair_ag_text)
        df["label_name"] = df["label"].map(lambda i: AG_LABELS[i])
        ag[part] = df[["orig_idx", "text", "label", "label_name", "text_raw"]]
    # After repair, 18 train rows become identical to test rows and 166 train rows
    # duplicate each other. Remove them from the pool so test stays truly unseen.
    pool = ag["train"].drop_duplicates("text")
    pool = pool[~pool["text"].isin(set(ag["test"]["text"]))]
    # Exactly n rows per class, then shuffled so classes are not in blocks.
    per_class = lambda df, n: (df.groupby("label").sample(n=n, random_state=SEED)
                               .sample(frac=1, random_state=SEED))
    train = per_class(pool, 7_500)                      # 30,000
    val = per_class(pool.drop(train.index), 500)        # 2,000, disjoint from train
    train_20k = per_class(train, 5_000)                 # 20,000, subset of train
    out["ag_train"], out["ag_val"], out["ag_train_20k"] = train, val, train_20k
    out["ag_test"] = ag["test"]

    # BBC: 98 articles appear twice verbatim and 37 more reuse the same body under a
    # reworded title. Keep the first copy only, otherwise the same text could land
    # in both n-gram train and test.
    raw = pd.read_parquet(RAW / "bbc_summary.parquet")
    raw["_parsed"] = raw["Articles"].map(parse_bbc_article)
    raw = raw[~raw["_parsed"].str[1].duplicated()]
    parsed = raw["_parsed"]
    bbc = pd.DataFrame({
        "id": range(len(raw)),
        "category": raw["File_path"].values,
        "title": [t for t, _ in parsed],
        "body": [b for _, b in parsed],
        "summary": raw["Summaries"].map(lambda t: fix_glued_sentences(html.unescape(t))).values,
    })
    bbc["label"] = bbc["category"].map(BBC_LABELS.index)
    # 80/20 article-level split for the n-gram LM (nb03), stratified by category.
    _, lm_test = _stratified_take(bbc, 0.8, label="category")
    bbc["lm_split"] = "train"
    bbc.loc[lm_test.index, "lm_split"] = "test"
    # Fixed 100-article sample for summarization ROUGE (nb09).
    bbc["summ_sample"] = bbc["id"].isin(bbc.sample(100, random_state=SEED)["id"])
    out["bbc"] = bbc

    # TweetEval: keep the official splits unchanged.
    for part in ["train", "val", "test"]:
        df = pd.read_parquet(RAW / f"tweet_{part}.parquet")
        df = df.rename(columns={"text": "text_raw"})
        df["text"] = df["text_raw"].map(repair_tweet_text)
        df = df[["text", "label", "text_raw"]]
        df["label_name"] = df["label"].map(lambda i: TWEET_LABELS[i])
        out[f"tweet_{part}"] = df

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df = df.reset_index(drop=True)
        out[name] = df
        df.to_parquet(SPLITS / f"{name}.parquet", index=False)
    return out


# ---- Load ------------------------------------------------------------------
def load(name: str, fast: bool = FAST) -> pd.DataFrame:
    """Load a split by name. With fast=True, keep FAST_PER_CLASS rows per class
    (AG and tweet splits only) so notebooks run in seconds."""
    df = pd.read_parquet(SPLITS / f"{name}.parquet")
    if fast and name.startswith(("ag_", "tweet_")):
        shuffled = df.sample(frac=1, random_state=SEED)
        df = shuffled.groupby("label").head(FAST_PER_CLASS).sort_index().reset_index(drop=True)
    return df
