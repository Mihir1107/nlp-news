# Viva question bank

Answers point to the code (`file:function`) and to measured results (`results/*.json`, see `docs/RESULTS.md`).
Numbers are from the final runs; if you re-run a notebook, re-check them in `docs/RESULTS.md`.

---

## A. Project and data

**1. What does NewsMind do, in one breath?**
It classifies news into 4 topics (AG News), summarizes BBC articles (TextRank, DistilBART), answers questions
about them (NewsBot), analyses tweet sentiment, and serves the main functions through a FastAPI REST API.
12 notebooks cover the syllabus; `src/` holds the from-scratch implementations.

**2. Why did you not use the `SetFit/bbc-news` dataset?**
Its text is lowercased with apostrophes stripped ("england s lead"). Capital letters are a main cue for NER,
and sentence splitting for TextRank also suffers. We used `gopalkalpande/bbc-news-summary`: the same
corpus in original case, with a reference summary for every article (docs/research_notes.md §3).

**3. What data cleaning did you have to do before any NLP?**
AG News had HTML artefacts (`#39;` = apostrophe in 29,696 train rows, `quot;`, `&lt;b&gt;`); BBC had
glued paragraphs (newlines became dots) and HTML entities; TweetEval had emoji stored as literal `\uXXXX`
escape text. `src/data.py` repairs these, and `tests/test_splits.py` checks the repairs.

**4. Why remove duplicates, and how many were there?**
98 BBC articles appeared twice and 37 more reused the same body under a new title, leaving 2,089 articles.
In AG News, after repair, 18 train rows became identical to test rows. Duplicates across train and test leak
answers into training and inflate scores.

**5. Why seed 42 and saved splits?**
So every notebook sees exactly the same rows, which makes results comparable and repeatable.
`test_rebuild_is_identical` rebuilds the splits and checks they are unchanged.

## B. Module I: preprocessing (nb01)

**6. Stemming vs lemmatization?**
Stemming chops suffixes by rule and can produce non-words (Porter: *studies* → *studi*, *organization* →
*organ*). Lemmatization returns the dictionary form using the POS tag (*was*/VERB → *be*, *better*/ADJ →
*good*). On AG train, lemmatizing cut the vocabulary from 47,472 to 41,292 words; Porter cut it to 35,762.

**7. Why does the lemmatizer need POS tags?**
WordNet assumes a noun by default, so *ran* stays *ran*. With the verb tag it returns *run*. We tag with NLTK's
perceptron tagger in `src/preprocess.py:lemmatize`.

**8. What bug did you find in WordNet lemmatization?**
It strips a final *s* whenever the rest is a dictionary word: *us* → *u*, *has* → *ha*, *boss* → *bos*.
It turned "US" into "u" in real headlines. `src/preprocess.py:_lemma` undoes those cases, and it is unit-tested.

**9. Why keep "not" when removing stop words?**
NLTK's 198-word list includes *not/no/nor*. Removing them flips meaning ("not a bad phone" → "bad phone").
We remove the rest; stop-word removal cut 32% of tokens but only 136 vocabulary entries (nb01).

**10. How is tweet preprocessing different?**
`clean(mode="tweet")` also replaces URLs and @mentions, splits CamelCase hashtags (#GameDay → game day),
maps emoji/emoticons to words (😂 → laugh, :) → happy) and shortens elongations (soooo → soo).

## C. Module II: BoW, TF-IDF, n-grams, POS, NER (nb02–nb04)

**11. Write the TF-IDF formula sklearn uses.**
`tfidf(t,d) = count(t,d) × idf(t)` with `idf(t) = ln((1+n)/(1+df(t))) + 1`, then each document row is
divided by its L2 length. `src/tfidf.py` implements this and `np.allclose` confirms it equals sklearn,
both on a toy corpus and on 2,000 AG documents.

**12. Why the "+1"s in sklearn's smoothed idf?**
The +1 inside the log acts as if one extra document contained every term, so there is never a division by
zero. The +1 outside keeps idf > 0, so a word found in every document is down-weighted but not deleted.

**13. Did TF-IDF beat Bag-of-Words?**
With unigrams yes: 0.9009 vs 0.8832 accuracy, +1.78 points, bootstrap CI [+1.30, +2.28]. With uni+bigrams
the gap (0.9018 vs 0.8982) is not significant, CI [−0.07, +0.76]. Bigrams help raw counts a lot, TF-IDF little.

**14. What do bigrams add?**
Short phrases ("prime minister", "oil price"). The cost is vocabulary growth: 21,326 → 107,993 features
with `min_df=2`.

**15. Why does smoothing exist?**
With maximum-likelihood counts, any n-gram never seen in training has probability 0, so the whole sentence
gets probability 0 and perplexity is infinite. In nb03 even a normal sentence gets −inf with the bigram
model because the pair *(taxes, next)* never occurred. Add-k gives every word a pseudo-count k.

**16. Laplace vs add-k: which is better here?**
Laplace (k = 1) spreads a full count over a 17,510-word vocabulary, so seen continuations lose most of their
mass. Bigram test perplexity: Laplace 2,251 vs add-k (k = 0.01) 551. With Laplace, even the bigram loses to
the unigram (968).

**17. Why can bigram perplexity beat unigram, while trigram loses?**
One word of context sharply narrows the next word (after "prime", "minister" has P ≈ 0.86), so bigram
perplexity drops from 964 to 551. But 63.9% of test trigrams never appear in training, so the trigram
falls back on uninformed smoothing mass: test 2,891, while training perplexity keeps falling (134).
That is overfitting caused by sparsity.

**18. What is perplexity?**
exp(−(1/N) Σ log P(wᵢ | history)) over all N predicted tokens (we count `</s>`). It is the effective
number of equally likely choices per word; lower is better. It is comparable only with the same vocabulary
and test set (`src/ngram.py:perplexity`).

**19. What is `<UNK>` for?**
Training words seen fewer than 2 times become `<UNK>`, so the model learns a probability for "some rare
word". An unseen test word maps to it instead of getting probability 0.

**20. How does n-gram text generation work?**
Start from `<s>` markers and repeatedly sample the next word in proportion to how often it followed the
current history, until `</s>` (`src/ngram.py:generate`). Unigrams give word salad, bigrams local sense,
trigrams near-copies of training sentences.

**21. POS tagging vs NER?**
POS gives every token a grammatical tag. NER finds multi-token spans that name things and types them (PERSON,
ORG, GPE…). Both are sequence labelling; NER usually uses BIO tags.

**22. How good was spaCy's small NER model on BBC?**
On 30 hand-checked sentences: 57 of 85 predicted entities exactly right (precision 0.671), 16 sentences with at
least one error, 3 entities missed. Errors: currency cut off ("220" for "$220m"), clubs vs people swapped,
metonymy (Newcastle = the club, tagged GPE).

## D. Module III: embeddings and sentiment (nb05, nb11)

**23. CBOW vs skip-gram?**
CBOW predicts the centre word from the averaged context: fast, good for frequent words. Skip-gram predicts
context words from the centre word, which gives rare words more updates. On AG, skip-gram + LR scored 0.8861
vs CBOW 0.8683.

**24. Word2Vec vs GloVe?**
Word2Vec is predictive: a shallow network trained on local windows. GloVe is count-based: it builds a
global co-occurrence matrix and fits vectors so that wᵢ·wⱼ + bᵢ + bⱼ ≈ log Xᵢⱼ. Both give one static
vector per word.

**25. Why do averaged embeddings lose information?**
The average is the same for any word order ("dog bites man" = "man bites dog"), negation scope is lost,
and decisive keywords are diluted by the other words. Both averaged models (0.886, 0.883) are below TF-IDF (0.902).

**26. Why did your Word2Vec fail king − man + woman?**
It saw only ~0.8M tokens of 2004 news, where "king" mostly means Cambodia's King Sihanouk, so it answered
*penh, norodom, sihanouk*. GloVe (6B tokens) answers *queen*. Analogies need large, varied text.

**27. Why do domain-matched embeddings matter?**
Vectors reflect how words are used in their training text. For news we used Wikipedia+Gigaword GloVe;
for tweets glove-twitter-100, which knows slang and informal spellings. The effect showed on tweets:
GloVe-Twitter averaging (macro-recall 0.608) beat TF-IDF (0.574), the opposite of the news result.

**28. What is special about TweetEval's test set?**
Label shift: negatives are 15.5% of train but 32.3% of test. A model that learns the training prior
under-predicts negatives (TF-IDF negative recall 0.383). The official metric is macro-recall.

**29. Did emoji handling help?**
Barely. The training tweets contain no emoji at all, while 6.6% of test tweets do. The only significant gain
was for TF-IDF, +0.0056 macro-F1. We showed it comes from emoticons like `:)` in the *training* tweets:
the gain also appears on test tweets with no emoji or emoticon (+0.0032, CI [+0.0007, +0.0054]).
Hashtag splitting had no significant effect.

**30. Why is the twitter-RoBERTa result not a fair comparison?**
It was pre-trained on tweets and fine-tuned for sentiment by the group that built TweetEval, possibly on
overlapping data. It is a ceiling (macro-recall 0.735), not a competitor.

## E. Module IV: deep learning and transformers (nb06–nb08)

**31. What do the LSTM gates do?**
Forget gate: what to erase from the cell state. Input gate: how much new candidate information to write.
Output gate: how much of the cell becomes the hidden state. c_t = f⊙c_{t−1} + i⊙g, h_t = o⊙tanh(c_t).

**32. Why do LSTMs reduce vanishing gradients?**
A plain RNN multiplies the gradient by the same recurrent matrix at every step, so it shrinks
exponentially. The LSTM's cell update is additive; with the forget gate near 1 the gradient passes through
almost unchanged. We also clip gradients (norm 5) against the exploding case.

**33. Why did GloVe initialisation matter so much for the BiLSTM?**
Random embeddings must learn word meaning from 30k headlines alone; GloVe starts with meaning already
encoded. Test accuracy 0.9066 vs 0.8626. 92% of the model's 3.0M parameters are the embedding table.

**34. Masked LM vs autoregressive LM?**
BERT hides 15% of tokens and predicts them from both sides, so it suits understanding tasks. GPT-2
predicts the next token from the left only (causal mask), so it can generate. In nb07, changing "money"
to "fish" to the *right* of the mask changes BERT's prediction from *bank* to *lake/river*.

**35. Why does fine-tuned DistilBERT beat averaged Word2Vec?**
Contextual vectors (each word represented using the whole sentence), 6 self-attention layers that model
order and interactions, and pre-training on billions of words, all adapted by fine-tuning. Accuracy 0.9174
vs 0.8861; the gap is significant.

**36. Then why is DistilBERT only ~1.5 points better than TF-IDF?**
AG News is keyword-heavy, so a linear model on word features is already strong. And about half of DistilBERT's
628 errors are shared by all six models, many of them label noise (a tennis story labelled "World"). The price
of those points: 268 MB vs 6 MB, and 24 minutes of training vs 2.5 seconds.

**37. What does "contextual embedding" mean? Show it.**
BERT gives the same word different vectors in different sentences. In nb07, "bank" in two money sentences
has cosine 0.812, and in two river sentences 0.815, but only 0.395–0.514 across senses. Word2Vec has one vector,
whose neighbours are all bank names.

**38. What is a hallucination?**
Fluent but false output. GPT-2, asked who won the 2004 Nobel Prize in Literature, said "Albert Einstein".
It maximises text likelihood, not truth. DistilBART also produced "$2.3bn Australian dollars" by merging two
currencies from the source (nb09).

**39. How do you extract subject–verb–object triples?**
spaCy dependency parse; for each verb take `nsubj`/`nsubjpass` children as subjects and `dobj`/`attr` (or the
`pobj` under a `prep`) as objects, expanded to noun chunks (nb08 `svo_triples`). It gave triples for 13 of 20
sentences, with visible errors: a PP-attachment mistake, and "Wal-Mart" split at the hyphen.

**40. Why can MiniLM find related text with no shared words?**
It was trained on sentence pairs to put paraphrases close together. "carmaker cuts jobs" found "GM, Ford cut
output" while TF-IDF returned a Whitehall job-cuts story. On whole articles MiniLM also scored higher
category precision@3 (0.938 vs 0.907).

## F. Module V: generation, summarization, QA, API (nb03, nb07, nb09, nb10, api/)

**41. How does TextRank work?**
Sentences → TF-IDF vectors → cosine-similarity graph → PageRank (power iteration, damping 0.85) → top-k
sentences in original order (`src/textrank.py`). Our PageRank equals networkx's in a unit test.

**42. Why is ROUGE flawed?**
It counts n-gram overlap with one reference, not meaning or correctness: a correct paraphrase scores low,
and a fluent sentence with one wrong number scores high. Here the references are 96.1% copied article
sentences, so ROUGE favours extractive systems: TextRank ROUGE-2 0.560, Lead-3 0.384, DistilBART 0.264.

**43. Extractive vs abstractive summarization?**
Extractive copies sentences: faithful but can be choppy. Abstractive (DistilBART) writes new text: fluent and
compact, but can add unsupported facts. We hand-checked 15 flagged DistilBART summaries: 2 real errors,
1 repetition, 12 false alarms.

**44. How does NewsBot answer a question?**
The TF-IDF retriever (our `src/tfidf.py`) finds the top 3 of 9,937 passages (120 words, 40-word overlap).
The DistilBERT-SQuAD reader scores every span as start + end logit (≤ 30 tokens) and the best one across
the passages wins. Below a confidence threshold of 0.48 (tuned in nb10) it says "I don't know".

**45. What part of the chatbot is rule-based?**
The intent router in `src/qa.py:NewsBot.chat`: regular expressions for "summarize article N", "who is
mentioned in article N" and "what topic is …". Only factual questions go to the learned retriever and reader.

**46. How good is NewsBot?**
On 30 answerable questions: retrieval hit@1 0.833 and hit@3 0.933; reader EM 0.767 and F1 0.802, rising to
0.933 EM when given the correct passage. Most errors are the reader picking a span from a wrong passage.
The threshold handles 75% of all 40 questions correctly, but it cannot catch confident wrong answers to
some unanswerable questions (a SQuAD 1.1 reader always answers).

**47. What does the API do with inputs beyond 512 tokens?**
DistilBERT can read at most 512 tokens, so `/classify` with `model=distilbert` truncates the rest and returns
`"truncated": true` (`api/app.py:classify`). DistilBART truncates at 1,024 tokens, with the same flag.
TF-IDF has no length limit. Requests over 10,000 characters are rejected with 422 before any model runs.

**48. How is input validated?**
Pydantic models: `text` 1–10,000 characters and not only whitespace, `model` ∈ {tfidf, distilbert},
`method` ∈ {textrank, distilbart}. FastAPI returns 422 with the reason; `tests/test_api.py` covers 5 invalid
cases.

**49. Why load models in `lifespan`, and why is DistilBART lazy?**
Loading once at start-up avoids reloading hundreds of MB on every request. DistilBART is 1.2 GB and rarely
used, so it loads on the first request that needs it. A warm-up call removes first-request latency (1.4 s →
~1 ms for TF-IDF).

**50. Why is TF-IDF the default classifier in the API?**
Within ~1.5 points of the best model, 6 MB, about 1 ms per request, and explainable (top features per class).
DistilBERT is opt-in when accuracy matters more than cost.

## G. Evaluation (CO5)

**51. Precision, recall, F1?**
Precision: of the items predicted as class X, the share that really are X. Recall: of the real X items, the
share we found. F1 = 2PR/(P+R), their harmonic mean, which is low if either is low.

**52. Macro vs micro F1?**
Macro-F1 averages per-class F1, so every class counts equally. Micro-F1 pools all decisions and, for
single-label classification, equals accuracy. On balanced AG News they are almost the same; on imbalanced,
shifted TweetEval macro is the honest choice.

**53. How do you know a difference between two models is real?**
Paired bootstrap: resample the test set 1,000 times and recompute the difference on the same resampled
documents each time. If the 95% interval excludes 0, the difference is larger than test-set noise. We used it
in nb02, nb11 and nb12.

**54. Why is the QA threshold score optimistic?**
The threshold was tuned and evaluated on the same 40 questions. A fair estimate needs a separate held-out
question set.

**55. What would you improve with more time?**
A SQuAD 2.0 reader that can say "no answer"; re-ranking QA spans by retrieval score as well; a larger spaCy
model (`en_core_web_trf`) for NER; Kneser–Ney smoothing for the n-gram LM; human evaluation of summaries
instead of ROUGE only.

## H. The web app (frontend/)

**56. How does the web app talk to the models?**
It is a static page (React) served by FastAPI at `/`. Each view calls one endpoint with `fetch` and JSON
(`frontend/src/api.js`): Topic → `POST /classify`, Summary → `POST /summarize`, Names & places →
`POST /entities`, Ask → `POST /ask`, results → `GET /results`. No model runs in the browser.

**57. Why build it with React and esbuild, and why are the fonts bundled?**
React keeps each view's state (inputs, results, the chat) in components, so switching tabs does not lose
anything; the five views stay mounted and only the visible one is shown. esbuild turns the JSX, CSS and font
files into `frontend/dist/` in one step (`build.mjs`). Bundled fonts mean the demo works without internet.

**58. How is TextRank's choice made visible?**
`/summarize` returns every sentence with its PageRank score and the indices of the kept ones
(`src/textrank.py:textrank_rank`). The page shows the whole article, marks the kept sentences in rank order
and draws a line for each sentence's score relative to the best one.

**59. How do the marks on the copy work?**
`components/ui.jsx:Marker` is a `<mark>` whose background is a solid gradient; Framer Motion animates its
`background-size` width from 0% to 100%, so the mark is drawn left to right. The "wash" variant is a tinted band
behind the words; the "pencil" variant is a 0.09em-high band at the baseline, which looks like an underline. `box-decoration-break: clone`
gives each wrapped line its own stroke. With the OS "reduce motion" setting on, the mark appears instantly.

**60. Are the numbers on the results page typed into the frontend?**
No. `GET /results` reads `results/*.json` on every request (`api/app.py:results`), including the few numbers
quoted in the explanatory text (NER precision, threshold, sample sizes).
