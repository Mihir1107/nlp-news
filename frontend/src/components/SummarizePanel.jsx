import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { api } from '../api.js'
import { ErrorNote, Marker, Meter, RunButton, Samples, Segmented, ms } from './ui.jsx'

const METHODS = [
  { value: 'textrank', label: 'TextRank (picks sentences)' },
  { value: 'distilbart', label: 'DistilBART (writes new text)' },
]

export default function SummarizePanel({ samples, facts }) {
  const [text, setText] = useState('')
  const [method, setMethod] = useState('textrank')
  const [k, setK] = useState(3)
  const [out, setOut] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Start with a real BBC article so the page is never empty. Runs when the samples arrive;
  // it only fills the box if the user has not typed anything yet.
  useEffect(() => {
    if (!text && samples?.articles?.length) setText(samples.articles[0].body)
  }, [samples])

  async function run(e) {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      setOut(await api('/summarize', { text, method, sentences: k }))
    } catch (err) {
      setError(err.message); setOut(null)
    } finally {
      setBusy(false)
    }
  }

  const articles = (samples?.articles || []).map((a) => ({ key: a.id, short: a.title, title: `BBC article ${a.id}`, text: a.body }))

  return (
    <div className="desk">
      <form className="sheet" onSubmit={run}>
        <label htmlFor="sum-text" className="field-label">The article</label>
        <textarea id="sum-text" className="reading" rows={10} value={text} required
                  placeholder="Paste a news article of a few paragraphs."
                  onChange={(e) => setText(e.target.value)} />
        <Samples label="Or open a story from the BBC archive:" items={articles} onPick={(a) => { setText(a.text); setOut(null) }} />
        <div className="controls">
          <Segmented id="sum" label="Method" options={METHODS} value={method} onChange={setMethod} />
          {method === 'textrank' && (
            <label className="slider">
              <span>{k} {k === 1 ? 'sentence' : 'sentences'}</span>
              <input type="range" min={1} max={6} value={k} onChange={(e) => setK(Number(e.target.value))} />
            </label>
          )}
          <RunButton busy={busy} busyText={method === 'distilbart' ? 'Writing… the first run loads a 1.2 GB model' : 'Ranking…'}>
            Summarize
          </RunButton>
        </div>
        <ErrorNote error={error} />
      </form>

      <div className="findings" aria-live="polite">
        <AnimatePresence mode="wait">
          {!out ? (
            <motion.p key="empty" className="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              TextRank reads the article as a web of similar sentences and keeps the ones the rest point to,
              marking its choices on the copy. DistilBART writes a summary of its own.
            </motion.p>
          ) : out.method === 'textrank' ? (
            <TextRankView key={`tr-${out.latency_ms}`} out={out} />
          ) : (
            <BartView key={`b-${out.latency_ms}`} out={out} facts={facts} />
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function TextRankView({ out }) {
  const max = Math.max(...out.scores)
  // rank of each selected sentence (0 = highest score) decides when its mark sweeps in
  const order = [...out.selected].sort((a, b) => out.scores[b] - out.scores[a])
  return (
    <motion.section className="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <p className="kicker">TextRank kept {out.selected.length} of {out.sentences.length} sentences, in {ms(out.latency_ms)}</p>
      <ol className="ranked-text">
        {out.sentences.map((s, i) => {
          const picked = out.selected.includes(i)
          const rank = order.indexOf(i)          // 0 = highest PageRank score
          return (
            <li key={i} className={picked ? 'picked' : 'dropped'}>
              {/* margin numeral: the sentence's rank, written in like an editor's note */}
              <motion.span className="rank" aria-label={picked ? `rank ${rank + 1}` : undefined}
                           initial={{ opacity: 0, x: -6 }} animate={{ opacity: picked ? 1 : 0, x: 0 }}
                           transition={{ delay: 0.25 + rank * 0.35, duration: 0.4 }}>
                {picked ? rank + 1 : ''}
              </motion.span>
              <span className="sentence">
                {picked ? <Marker delay={0.2 + rank * 0.35}>{s}</Marker> : s}
              </span>
              <span className="score" title={`PageRank score ${out.scores[i]}`}>
                <Meter value={out.scores[i] / max} height={2} color={picked ? 'var(--pencil)' : 'var(--rule)'} delay={i * 0.02} />
              </span>
            </li>
          )
        })}
      </ol>
      <p className="note">Numbers in the margin: the kept sentences in order of importance. The rule beside each
        sentence is its PageRank score relative to the best one.</p>
    </motion.section>
  )
}

function BartView({ out, facts }) {
  const words = out.summary.split(' ')
  return (
    <motion.section className="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <p className="kicker">DistilBART wrote this summary, in {ms(out.latency_ms)}</p>
      <p className="generated dropcap">
        {words.map((w, i) => (
          <motion.span key={i} initial={{ opacity: 0, y: 6, filter: 'blur(3px)' }}
                       animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                       transition={{ delay: i * 0.025, duration: 0.3 }}>
            {w}{' '}
          </motion.span>
        ))}
      </p>
      <p className="note">
        Generated text can state things the article does not
        {facts && ` (we found this in ${facts.bart_real_errors} of ${facts.bart_checked} summaries we checked)`}, so check names
        and numbers against the article.
        {out.truncated && ' The article was longer than 1,024 tokens, so only the beginning was read.'}
      </p>
    </motion.section>
  )
}
