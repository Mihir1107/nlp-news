"""Text preprocessing. Every step is a small function so nb01 can show each one alone.

clean(text, mode="news"):  lowercase -> tokenize -> drop punctuation -> drop stop words -> lemmatize
clean(text, mode="tweet"): first handle URLs, @mentions, hashtags, emoji and elongated words,
                           then the same steps as news mode.
clean() returns a space-joined string, so its output can go straight into sklearn vectorizers.
"""
import re
import string

import src.utils  # noqa: F401  (points NLTK at data/nltk_data)
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK's list has 198 words and includes "not", "no", "nor". Removing those flips
# meaning ("not good" -> "good"), so we keep negations.
NEGATIONS = {"no", "not", "nor", "never", "n't"}
# "'s" (possessive, "Nestle's" -> "nestle", "'s") is not in NLTK's list, so we add it.
STOP_WORDS = (set(stopwords.words("english")) - NEGATIONS) | {"'s"}

_lemmatizer = WordNetLemmatizer()

# Emoji and emoticons mapped to ordinary words. Words carry sentiment the model can
# learn from training text; TweetEval's train split contains no emoji at all.
EMOJI_WORDS = {
    "😂": "laugh", "🤣": "laugh", "😁": "happy", "😊": "happy", "😀": "happy", "🙂": "happy",
    "😍": "love", "❤": "love", "💙": "love", "💕": "love", "😘": "love", "🥰": "love",
    "😭": "cry", "😢": "sad", "😩": "tired", "😞": "sad", "💔": "heartbroken",
    "😡": "angry", "😠": "angry", "🙄": "annoyed", "🤔": "thinking", "😒": "annoyed",
    "👍": "good", "👌": "good", "👏": "applause", "🙌": "celebrate", "🙏": "thanks", "🔥": "fire",
    ":)": "happy", ":-)": "happy", ":D": "laugh", ";)": "wink", ":(": "sad", ":-(": "sad", ":P": "playful",
}
_EMOTICON = re.compile(r"(?<!\w)(:-?\)|:-?\(|:D|;\)|:P)(?!\w)")
_OTHER_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")  # incl. skin tones


# ---- Individual steps ------------------------------------------------------
def lowercase(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list[str]:
    """NLTK's Treebank-style tokenizer: splits "don't" -> "do", "n't" and "$10" -> "$", "10"."""
    return word_tokenize(text)


def regex_tokenize(text: str) -> list[str]:
    """Simplest tokenizer: runs of letters/digits, keeping inner apostrophes ("don't")."""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text)


def remove_punctuation(tokens: list[str]) -> list[str]:
    """Drop tokens made only of punctuation ("." "--" "''"); keep "n't" and "u.s."."""
    return [t for t in tokens if not all(ch in string.punctuation + "‘’“”…" for ch in t)]


def remove_stopwords(tokens: list[str], stop_words: set[str] = STOP_WORDS) -> list[str]:
    return [t for t in tokens if t.lower() not in stop_words]


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank tag (NN, VBD, JJ, RB...) to a WordNet POS letter."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(treebank_tag[0], wordnet.NOUN)


def _lemma(tok: str, pos: str) -> str:
    lemma = _lemmatizer.lemmatize(tok, pos)
    # WordNet quirk: it strips a final "s" whenever the rest is in its dictionary,
    # so "us" -> "u", "has" -> "ha", "boss" -> "bos". Undo those cases.
    if lemma == tok[:-1] and tok.endswith("s") and (len(lemma) <= 2 or tok.endswith("ss")):
        return tok
    return lemma


def lemmatize(tokens: list[str]) -> list[str]:
    """WordNet lemmatizer with POS tags: "running"/VBG -> "run", "better"/JJR -> "good"."""
    return [_lemma(tok, _wordnet_pos(tag)) for tok, tag in nltk.pos_tag(tokens)]


# ---- Tweet-specific steps --------------------------------------------------
def replace_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " url ", text)


def replace_mentions(text: str) -> str:
    return re.sub(r"@\w+", " user ", text)  # TweetEval already anonymises to "@user"


def split_hashtags(text: str) -> str:
    """'#HappyBirthdayRemus' -> 'Happy Birthday Remus' (split on camel case, drop '#')."""
    return re.sub(r"#(\w+)", lambda m: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m[1]), text)


def replace_emoji(text: str) -> str:
    """Known emoji/emoticons -> words; any other emoji is removed."""
    text = _EMOTICON.sub(lambda m: f" {EMOJI_WORDS[m[1]]} ", text)
    for emo, word in EMOJI_WORDS.items():
        if not emo.isascii():
            text = text.replace(emo, f" {word} ")
    return _OTHER_EMOJI.sub(" ", text)


def shorten_elongated(text: str) -> str:
    """'soooo gooood' -> 'soo good'... keeps 2 letters, so 'soo' still differs from 'so'."""
    return re.sub(r"(\w)\1{2,}", r"\1\1", text)


# ---- Full pipeline ---------------------------------------------------------
def clean(text: str, mode: str = "news", remove_stop: bool = True, lemma: bool = True,
          handle_emoji: bool = True, handle_hashtags: bool = True) -> str:
    """Run the whole pipeline and return cleaned tokens joined by spaces.

    handle_emoji / handle_hashtags only matter in tweet mode (used for the nb11 ablation).
    When switched off, '#' and emoji are simply stripped like other punctuation.
    """
    if mode not in ("news", "tweet"):
        raise ValueError(f"mode must be 'news' or 'tweet', got {mode!r}")
    if mode == "tweet":
        text = replace_mentions(replace_urls(text))
        text = split_hashtags(text) if handle_hashtags else text.replace("#", " ")
        text = replace_emoji(text) if handle_emoji else _OTHER_EMOJI.sub(" ", text)
        text = shorten_elongated(text)
    tokens = remove_punctuation(tokenize(lowercase(text)))
    if remove_stop:
        tokens = remove_stopwords(tokens)
    if lemma:
        tokens = lemmatize(tokens)
    return " ".join(tokens)
