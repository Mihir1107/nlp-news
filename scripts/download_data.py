"""Recreate everything under data/ (gitignored) and build the shared splits.

    python scripts/download_data.py                 # everything (~4 GB)
    python scripts/download_data.py --skip-models   # datasets, NLTK, spaCy, GloVe only

Every network step is retried because downloads can fail mid-way.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import DATA, RAW  # noqa: E402  (sets cache env vars before HF imports)

NLTK_RESOURCES = ["punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"]
GLOVE = ["glove-wiki-gigaword-100", "glove-twitter-100"]
# Hub id -> transformers Auto class used to load it (downloads only the files needed).
HF_MODELS = {
    "distilbert/distilbert-base-uncased": "AutoModelForSequenceClassification",
    "sshleifer/distilbart-cnn-12-6": "AutoModelForSeq2SeqLM",
    "distilbert/distilbert-base-cased-distilled-squad": "AutoModelForQuestionAnswering",
    "openai-community/gpt2": "AutoModelForCausalLM",
    "google-bert/bert-base-uncased": "AutoModelForMaskedLM",
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "AutoModelForSequenceClassification",
}
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retry(fn, what, tries=6, wait=15):
    """Call fn(); on any error wait and try again."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # network errors come in many types
            print(f"  ! {what} failed ({type(e).__name__}), attempt {attempt}/{tries}")
            if attempt == tries:
                raise
            time.sleep(wait)


def get_datasets():
    from datasets import load_dataset
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [  # (hub id, config, {hub split: our file name})
        ("fancyzhx/ag_news", None, {"train": "ag_train", "test": "ag_test"}),
        ("gopalkalpande/bbc-news-summary", None, {"train": "bbc_summary"}),
        ("SetFit/bbc-news", None, {"train": "bbc_setfit_train", "test": "bbc_setfit_test"}),
        ("cardiffnlp/tweet_eval", "sentiment",
         {"train": "tweet_train", "validation": "tweet_val", "test": "tweet_test"}),
    ]
    for hub_id, config, files in jobs:
        if all((RAW / f"{f}.parquet").exists() for f in files.values()):
            print(f"  = {hub_id} already in data/raw")
            continue
        ds = retry(lambda: load_dataset(hub_id, config), hub_id)
        for split, fname in files.items():
            ds[split].to_parquet(str(RAW / f"{fname}.parquet"))
            print(f"  + {hub_id}[{split}] -> data/raw/{fname}.parquet ({len(ds[split])} rows)")


def get_nltk():
    import nltk
    for r in NLTK_RESOURCES:
        retry(lambda: nltk.download(r, download_dir=str(DATA / "nltk_data"), quiet=True,
                                    raise_on_error=True), r)
    print("  + NLTK:", ", ".join(NLTK_RESOURCES))


def get_spacy():
    try:
        import en_core_web_sm  # noqa: F401
        print("  = spaCy en_core_web_sm already installed")
    except ImportError:
        retry(lambda: subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                                     check=True), "en_core_web_sm")


def get_glove():
    import gensim.downloader as api
    for name in GLOVE:
        retry(lambda: api.load(name, return_path=True), name)
        print(f"  + {name}")


def get_models():
    import transformers
    from sentence_transformers import SentenceTransformer
    for hub_id, cls in HF_MODELS.items():
        retry(lambda: transformers.AutoTokenizer.from_pretrained(hub_id), hub_id + " tokenizer")
        retry(lambda: getattr(transformers, cls).from_pretrained(hub_id), hub_id)
        print(f"  + {hub_id}")
    retry(lambda: SentenceTransformer(SENTENCE_MODEL), SENTENCE_MODEL)
    print(f"  + {SENTENCE_MODEL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true", help="skip HF model weights (~3.3 GB)")
    ap.add_argument("--skip-glove", action="store_true", help="skip GloVe vectors (~540 MB)")
    args = ap.parse_args()

    print("[1] datasets"); get_datasets()
    print("[2] NLTK data"); get_nltk()
    print("[3] spaCy model"); get_spacy()
    if not args.skip_glove:
        print("[4] GloVe"); get_glove()
    if not args.skip_models:
        print("[5] HF models"); get_models()
    print("[6] splits")
    from src.data import build_splits
    for name, df in build_splits().items():
        print(f"  {name:14s} {len(df):>6} rows")


if __name__ == "__main__":
    main()
