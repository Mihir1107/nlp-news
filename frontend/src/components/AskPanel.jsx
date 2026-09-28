import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { api } from '../api.js'
import { ErrorNote, Marker, Meter, ms } from './ui.jsx'

// How each answer was produced. Only "question" goes through the retriever and reader;
// the other three are matched by regular expressions in src/qa.py (NewsBot.chat).
const HOW = {
  question: 'Found in the archive by the reader',
  summarize: 'Rule-based command: summarize article N',
  entities: 'Rule-based command: who is mentioned in article N',
  topic: 'Rule-based command: what topic is …',
}

export default function AskPanel({ samples, facts }) {
  const [entries, setEntries] = useState([])   // the interview so far: questions and answers
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const end = useRef(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [entries, busy])

  async function send(question) {
    if (!question.trim() || busy) return
    setEntries((e) => [...e, { role: 'q', text: question, id: Date.now() }])
    setDraft(''); setBusy(true); setError('')
    try {
      const r = await api('/ask', { question })
      setEntries((e) => [...e, { role: 'a', ...r, id: Date.now() + 1 }])
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const suggestions = [...(samples?.questions || []), ...(samples?.commands || [])]

  return (
    <div className="interview-layout">
      <div className="interview" aria-live="polite">
        {entries.length === 0 && (
          <p className="empty standfirst">
            Put a question to the archive of {facts ? facts.n_articles.toLocaleString('en') : ''} BBC stories from
            2004–05. NewsBot finds the three closest passages, reads them with a question-answering model, and
            says "I don't know" when it is not sure enough. A few commands, such as "summarize article 1744",
            are handled by simple rules instead.
          </p>
        )}
        <AnimatePresence initial={false}>
          {entries.map((m) => (
            <motion.div key={m.id} layout className={m.role === 'q' ? 'qa q' : 'qa a'}
                        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}>
              <span className="qa-mark" aria-hidden="true">{m.role === 'q' ? 'Q.' : 'A.'}</span>
              {m.role === 'q' ? <p className="question">{m.text}</p> : <Answer m={m} threshold={facts?.qa_threshold} />}
            </motion.div>
          ))}
          {busy && (
            <motion.div key="reading" className="qa a reading-note" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <span className="qa-mark" aria-hidden="true">A.</span>
              <p className="muted"><em>Reading the archive</em><span className="ellipsis" /></p>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={end} />

        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(draft) }}>
          <label htmlFor="ask-input" className="qa-mark">Q.</label>
          <input id="ask-input" value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={1000}
                 placeholder="Who is the head of Tiga?" autoComplete="off" />
          <button type="submit" className="run" disabled={busy || !draft.trim()}><span className="run-label">Ask</span></button>
        </form>
        <ErrorNote error={error} />
      </div>

      <aside className="question-desk" aria-label="Suggested questions">
        <p className="kicker">From the question desk</p>
        <ul>
          {suggestions.map((q) => (
            <li key={q}><button type="button" className="link" onClick={() => send(q)} disabled={busy}>{q}</button></li>
          ))}
        </ul>
      </aside>
    </div>
  )
}

function Answer({ m, threshold }) {
  const [open, setOpen] = useState(false)
  const unsure = m.intent === 'question' && m.answer !== m.best_guess
  return (
    <div className="answer-block">
      {m.intent === 'question' && !unsure
        ? <blockquote className="pull"><Marker variant="pencil" color="var(--pencil)" delay={0.3}>{m.answer}</Marker></blockquote>
        : <p className={m.intent === 'question' ? 'pull quiet' : 'answer-text'}>{m.answer}</p>}

      <p className="byline">
        <em>{unsure ? 'Not confident enough to answer' : HOW[m.intent] || m.intent}</em>
        {m.title && <>, from “{m.title}” (article {m.article_id}{m.category ? `, ${m.category}` : ''})</>}
        , in {ms(m.latency_ms)}.
      </p>

      {m.intent === 'question' && (
        <>
          <div className="confidence">
            <span>Confidence {Math.round(m.confidence * 100)}%</span>
            <div className="meter-wrap">
              <Meter value={m.confidence} height={3} color={unsure ? 'var(--muted)' : 'var(--pencil)'} delay={0.1} />
              {threshold !== undefined && (
                <span className="threshold" style={{ left: `${threshold * 100}%` }}
                      title={`Below ${Math.round(threshold * 100)}% NewsBot says it doesn't know`} />
              )}
            </div>
          </div>
          {unsure && <p className="note">Its best guess was "{m.best_guess}", which falls below the threshold.</p>}
          <button type="button" className="link source-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? 'Hide the source passage' : 'Show the source passage'}
          </button>
          <AnimatePresence>
            {open && (
              <motion.blockquote className="passage" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                                 exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.35 }}>
                <p>
                  …{m.passage.slice(0, m.answer_start)}
                  <Marker delay={0.25} color={unsure ? 'var(--wash-slate)' : 'var(--wash-pencil)'}>
                    {m.passage.slice(m.answer_start, m.answer_end)}
                  </Marker>
                  {m.passage.slice(m.answer_end)}…
                </p>
              </motion.blockquote>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  )
}
