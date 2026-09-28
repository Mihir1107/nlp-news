import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { api, TOPIC_COLOR } from '../api.js'
import { ErrorNote, Marker, Meter, RunButton, Samples, Segmented, ms, shorten } from './ui.jsx'

const MODELS = [
  { value: 'tfidf', label: 'TF-IDF' },
  { value: 'distilbert', label: 'DistilBERT' },
  { value: 'both', label: 'Compare both' },
]
const NAMES = { tfidf: 'TF-IDF + logistic regression', distilbert: 'Fine-tuned DistilBERT' }

export default function ClassifyPanel({ samples }) {
  const [text, setText] = useState('')
  const [model, setModel] = useState('tfidf')
  const [results, setResults] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function run(e) {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      const which = model === 'both' ? ['tfidf', 'distilbert'] : [model]
      setResults(await Promise.all(which.map((m) => api('/classify', { text, model: m }))))
    } catch (err) {
      setError(err.message); setResults([])
    } finally {
      setBusy(false)
    }
  }

  const headlines = (samples?.headlines || []).map((h, i) => ({ key: i, short: shorten(h.text, 6), title: `AG News test set, labelled ${h.label}`, text: h.text }))

  return (
    <div className="desk">
      <form className="sheet" onSubmit={run}>
        <label htmlFor="classify-text" className="field-label">News text</label>
        <textarea id="classify-text" className="reading" rows={8} value={text} required
                  placeholder="Paste a headline or a paragraph from a news story."
                  onChange={(e) => setText(e.target.value)} />
        <Samples label="Real AG News test headlines" items={headlines} onPick={(h) => setText(h.text)} />
        <div className="controls">
          <Segmented id="clf" label="Model" options={MODELS} value={model} onChange={setModel} />
          <RunButton busy={busy} busyText="Reading…">Find the topic</RunButton>
        </div>
        <ErrorNote error={error} />
      </form>

      <div className="findings" aria-live="polite">
        <AnimatePresence mode="popLayout">
          {results.length === 0 ? (
            <motion.p key="empty" className="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              The topic appears here: World, Sports, Business or Sci/Tech, the four AG News classes both
              models were trained on.
            </motion.p>
          ) : (
            <motion.div key={results.map((r) => r.model + r.label + r.latency_ms).join()} className={results.length > 1 ? 'compare' : ''}
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {results.map((r, i) => <TopicResult key={r.model} r={r} delay={i * 0.15} />)}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function TopicResult({ r, delay }) {
  const ranked = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1])
  return (
    <section className="result">
      <p className="result-model">{NAMES[r.model]}</p>
      <h2 className="verdict">
        <Marker color={TOPIC_COLOR[r.label]} delay={delay + 0.1}>{r.label}</Marker>
      </h2>
      <p className="verdict-sub">{Math.round(r.confidence * 100)}% confident, answered in {ms(r.latency_ms)}</p>
      <ul className="bars">
        {ranked.map(([label, p], i) => (
          <li key={label}>
            <span className="bar-label">{label}</span>
            <Meter value={p} color={TOPIC_COLOR[label]} delay={delay + 0.2 + i * 0.06} />
            <span className="bar-value">{(p * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      {r.truncated && <p className="note">This text is longer than 512 tokens, so DistilBERT read only the beginning.</p>}
    </section>
  )
}
