import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { api } from '../api.js'
import { ErrorNote, Marker, Meter, ms } from './ui.jsx'

const INTENT = {
  question: 'Answered from the archive',
  summarize: 'Rule-based: summarize article N',
  entities: 'Rule-based: people in article N',
  topic: 'Rule-based: topic',
}
const BBC_COLOR = { business: 'var(--yellow)', entertainment: 'var(--lilac)', politics: 'var(--sky)', sport: 'var(--mint)', tech: 'var(--pink)' }

export default function AskPanel({ samples, facts }) {
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const end = useRef(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, busy])

  async function send(question) {
    if (!question.trim() || busy) return
    setMessages((m) => [...m, { role: 'you', text: question, id: Date.now() }])
    setDraft(''); setBusy(true); setError('')
    try {
      const r = await api('/ask', { question })
      setMessages((m) => [...m, { role: 'bot', ...r, id: Date.now() + 1 }])
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const suggestions = [...(samples?.questions || []), ...(samples?.commands || [])]

  return (
    <div className="chat">
      <div className="chat-log" aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-intro">
            <p className="empty">
              Ask about anything in the {facts ? facts.n_articles.toLocaleString('en') : ''} BBC articles from 2004–05. NewsBot finds the three most similar
              passages, reads them with a question-answering model, and says "I don't know" when it is not sure.
              Commands like "summarize article 1744" are handled by simple rules instead.
            </p>
          </div>
        )}
        <AnimatePresence initial={false}>
          {messages.map((m) => (
            <motion.div key={m.id} layout className={`bubble ${m.role}`}
                        initial={{ opacity: 0, y: 14, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                        transition={{ type: 'spring', stiffness: 380, damping: 30 }}>
              {m.role === 'you' ? m.text : <BotAnswer m={m} threshold={facts?.qa_threshold} />}
            </motion.div>
          ))}
          {busy && (
            <motion.div key="typing" className="bubble bot typing" aria-label="NewsBot is reading"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {[0, 1, 2].map((i) => (
                <motion.span key={i} className="dot" animate={{ y: [0, -5, 0] }}
                             transition={{ repeat: Infinity, duration: 0.8, delay: i * 0.15 }} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={end} />
      </div>

      <div className="suggestions">
        {suggestions.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => send(q)} disabled={busy}>{q}</button>
        ))}
      </div>
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(draft) }}>
        <label htmlFor="ask-input" className="visually-hidden">Question</label>
        <input id="ask-input" value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={1000}
               placeholder="Who is the head of Tiga?" autoComplete="off" />
        <button type="submit" className="run" disabled={busy || !draft.trim()}><span className="run-label">Ask</span></button>
      </form>
      <ErrorNote error={error} />
    </div>
  )
}

function BotAnswer({ m, threshold }) {
  const [open, setOpen] = useState(false)
  const unsure = m.intent === 'question' && m.answer !== m.best_guess
  return (
    <div className="bot-answer">
      <span className="intent">{unsure ? 'Not confident enough to answer' : INTENT[m.intent] || m.intent}</span>
      <p className="answer">
        {m.intent === 'question' && !unsure ? <Marker delay={0.2}>{m.answer}</Marker> : m.answer}
      </p>

      {m.intent === 'question' && (
        <>
          <div className="confidence">
            <span>Confidence {Math.round(m.confidence * 100)}%</span>
            <div className="meter-wrap">
              <Meter value={m.confidence} height={6} color={unsure ? 'var(--muted)' : 'var(--ink)'} delay={0.1} />
              {threshold !== undefined && (
                <span className="threshold" style={{ left: `${threshold * 100}%` }}
                      title={`Below ${Math.round(threshold * 100)}% NewsBot says it doesn't know`} />
              )}
            </div>
          </div>
          {unsure && <p className="note">Best guess was "{m.best_guess}", but that is below the confidence threshold.</p>}
          <button type="button" className="source-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
            <span className="swatch" style={{ background: BBC_COLOR[m.category] }} />
            {m.title} <span className="muted">(article {m.article_id}, {m.category})</span>
          </button>
          <AnimatePresence>
            {open && (
              <motion.p className="passage" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3 }}>
                …{m.passage.slice(0, m.answer_start)}
                <Marker delay={0.25} color={unsure ? 'var(--rule)' : 'var(--yellow)'}>{m.passage.slice(m.answer_start, m.answer_end)}</Marker>
                {m.passage.slice(m.answer_end)}…
              </motion.p>
            )}
          </AnimatePresence>
        </>
      )}
      {m.intent !== 'question' && m.title && <p className="muted small">Article {m.article_id}: {m.title}</p>}
      <p className="muted small">{ms(m.latency_ms)}</p>
    </div>
  )
}
