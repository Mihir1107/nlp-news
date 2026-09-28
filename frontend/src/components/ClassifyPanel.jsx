import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { api, TOPIC_COLOR } from '../api.js'
import { ErrorNote, Marker, Meter, RunButton, Samples, Segmented, ms, shorten } from './ui.jsx'

const MODELS = [
  { value: 'tfidf', label: 'TF-IDF' },
  { value: 'distilbert', label: 'DistilBERT' },
  { value: 'both', label: 'Compare both' },
]
const NAMES = { tfidf: 'TF-IDF', distilbert: 'DistilBERT' }

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
        <label htmlFor="classify-text" className="field-label">The copy</label>
        <textarea id="classify-text" className="reading" rows={5} value={text} required
                  placeholder="Paste a headline or a paragraph from a news story."
                  onChange={(e) => setText(e.target.value)} />
        <Samples label="Or take a headline from the AG News test set:" items={headlines} onPick={(h) => setText(h.text)} />
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
              Both models were trained to file a story under one of four desks: World, Sports, Business or
              Sci/Tech. Paste a headline and choose who reads it.
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
  const color = TOPIC_COLOR[r.label]
  return (
    <section className="result">
      <p className="kicker">{NAMES[r.model]} files it under</p>
      <h2 className="verdict">
        {/* the word is "typeset" in with a clip reveal, then underlined in its section colour */}
        <motion.span className="verdict-word" initial={{ clipPath: 'inset(0 100% 0 0)' }}
                     animate={{ clipPath: 'inset(0 0% 0 0)' }}
                     transition={{ duration: 0.7, delay, ease: [0.65, 0, 0.35, 1] }}>
          <Marker variant="pencil" color={color.ink} delay={delay + 0.55}>{r.label}</Marker>
        </motion.span>
      </h2>
      <p className="verdict-sub">
        {Math.round(r.confidence * 100)} per cent sure, in {ms(r.latency_ms)}
      </p>
      <ul className="bars">
        {ranked.map(([label, p], i) => (
          <li key={label} className={label === r.label ? 'lead' : ''}>
            <span className="bar-label">{label}</span>
            <Meter value={p} color={TOPIC_COLOR[label].ink} delay={delay + 0.3 + i * 0.07} />
            <span className="bar-value">{(p * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      {r.truncated && <p className="note">This text is longer than 512 tokens, so DistilBERT read only the beginning.</p>}
    </section>
  )
}
