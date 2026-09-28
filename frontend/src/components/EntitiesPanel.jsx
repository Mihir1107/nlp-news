import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { api, entityKind } from '../api.js'
import { ErrorNote, Marker, RunButton, Samples, ms } from './ui.jsx'

export default function EntitiesPanel({ samples, facts }) {
  const [text, setText] = useState('')
  const [out, setOut] = useState(null)       // { text, entities, latency_ms }
  const [focus, setFocus] = useState(null)   // kind the reader clicked in the legend
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function run(e) {
    e.preventDefault()
    setBusy(true); setError(''); setFocus(null)
    try {
      const r = await api('/entities', { text })
      setOut({ ...r, text })
    } catch (err) {
      setError(err.message); setOut(null)
    } finally {
      setBusy(false)
    }
  }

  // First paragraph of each sample article: long enough to contain several names.
  const items = (samples?.articles || []).map((a) => ({
    key: a.id, short: a.title, title: `BBC article ${a.id}`,
    text: a.body.split(/(?<=\.)\s+/).slice(0, 4).join(' '),
  }))

  return (
    <div className="desk">
      <form className="sheet" onSubmit={run}>
        <label htmlFor="ent-text" className="field-label">Text</label>
        <textarea id="ent-text" className="reading" rows={9} value={text} required
                  placeholder="Paste a few sentences that mention people, organisations and places."
                  onChange={(e) => setText(e.target.value)} />
        <Samples label="Opening paragraphs" items={items} onPick={(a) => setText(a.text)} />
        <div className="controls">
          <RunButton busy={busy} busyText="Reading…">Find names and places</RunButton>
        </div>
        <ErrorNote error={error} />
      </form>

      <div className="findings" aria-live="polite">
        <AnimatePresence mode="wait">
          {!out ? (
            <motion.p key="empty" className="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              spaCy's small English model marks every name it finds.
              {facts && ` When we checked ${facts.ner_checked} of its entities in BBC text by hand, ${Math.round(facts.ner_precision * 100)}% were exactly right, so expect some mistakes.`}
            </motion.p>
          ) : (
            <motion.section key={out.latency_ms} className="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <Legend entities={out.entities} focus={focus} setFocus={setFocus} />
              <p className="reading annotated">{annotate(out, focus)}</p>
              <p className="note">{out.entities.length} entities found in {ms(out.latency_ms)}. Click a colour to show one kind.</p>
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function Legend({ entities, focus, setFocus }) {
  const counts = {}
  entities.forEach((e) => {
    const [kind, color] = entityKind(e.label)
    counts[kind] = counts[kind] || { n: 0, color }
    counts[kind].n += 1
  })
  return (
    <div className="legend">
      {Object.entries(counts).map(([kind, { n, color }]) => (
        <button key={kind} type="button" aria-pressed={focus === kind}
                className={`legend-item ${focus && focus !== kind ? 'faded' : ''}`}
                onClick={() => setFocus(focus === kind ? null : kind)}>
          <span className="swatch" style={{ background: color }} />{kind} <span className="count">{n}</span>
        </button>
      ))}
    </div>
  )
}

/** Rebuild the text with each entity wrapped in a Marker, sweeping in one after another. */
function annotate({ text, entities }, focus) {
  const parts = []
  let at = 0
  entities.forEach((e, i) => {
    const [kind, color] = entityKind(e.label)
    parts.push(text.slice(at, e.start))
    parts.push(
      <Marker key={i} color={color} delay={0.1 + Math.min(i, 20) * 0.07}
              className={focus && focus !== kind ? 'dim' : ''} title={`${kind} (spaCy label ${e.label})`}>
        {text.slice(e.start, e.end)}<span className="tag">{kind}</span>
      </Marker>,
    )
    at = e.end
  })
  parts.push(text.slice(at))
  return parts
}
