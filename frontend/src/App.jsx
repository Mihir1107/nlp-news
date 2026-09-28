import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { motion } from 'framer-motion'
import { api } from './api.js'
import Lamp from './components/Lamp.jsx'
import ClassifyPanel from './components/ClassifyPanel.jsx'
import SummarizePanel from './components/SummarizePanel.jsx'
import EntitiesPanel from './components/EntitiesPanel.jsx'
import AskPanel from './components/AskPanel.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'

// Sections of the "magazine". Each one calls one API endpoint.
const SECTIONS = [
  { id: 'topic', label: 'Topic' },
  { id: 'summary', label: 'Summary' },
  { id: 'names', label: 'Names & places' },
  { id: 'ask', label: 'Ask the archive' },
  { id: 'results', label: 'The numbers' },
]
const EASE = [0.65, 0, 0.35, 1]

// The theme is decided before React starts (inline script in index.html), so read it from <html>.
const initialTheme = () => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')

export default function App() {
  const [tab, setTab] = useState('topic')
  const [theme, setTheme] = useState(initialTheme)
  const lamp = useRef(null)
  const [health, setHealth] = useState(null)     // null = checking, false = unreachable
  const [samples, setSamples] = useState(null)
  const [results, setResults] = useState(null)

  // One round of requests when the page opens: API status, example inputs, evaluation numbers.
  useEffect(() => {
    api('/health').then(setHealth).catch(() => setHealth(false))
    api('/samples').then(setSamples).catch(() => {})
    api('/results').then(setResults).catch(() => {})
  }, [])

  // Light or put out the lamp. The new edition spreads out from the lamp (a View Transition whose
  // circle starts at the lantern); browsers without View Transitions, or with reduced motion, switch at once.
  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    const r = lamp.current.getBoundingClientRect()
    const root = document.documentElement
    root.style.setProperty('--vt-x', `${r.left + r.width / 2}px`)
    root.style.setProperty('--vt-y', `${r.top + r.height * 0.22}px`)
    const apply = () => {
      root.dataset.theme = next
      flushSync(() => setTheme(next))  // render the lit/unlit lamp inside the transition snapshot
    }
    try { localStorage.setItem('newsmind-theme', next) } catch { /* private mode: not remembered */ }
    if (!document.startViewTransition || matchMedia('(prefers-reduced-motion: reduce)').matches) apply()
    else document.startViewTransition(apply)
  }

  // Arrow keys move between sections (standard tablist behaviour).
  function onTabKey(e) {
    const i = SECTIONS.findIndex((t) => t.id === tab)
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!step) return
    const next = SECTIONS[(i + step + SECTIONS.length) % SECTIONS.length].id
    setTab(next)
    document.getElementById(`tab-${next}`)?.focus()
  }

  const facts = results?.facts
  const panels = {
    topic: <ClassifyPanel samples={samples} />,
    summary: <SummarizePanel samples={samples} facts={facts} />,
    names: <EntitiesPanel samples={samples} facts={facts} />,
    ask: <AskPanel samples={samples} facts={facts} />,
    results: <ResultsPanel results={results} />,
  }

  return (
    <div className="page">
      {/* At night the lamp throws a warm pool of light over the masthead. */}
      {theme === 'dark' && <motion.div className="lamplight" aria-hidden="true"
                                       initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.2 }} />}
      {/* The one orchestrated moment: the double rule is drawn across, then the nameplate settles. */}
      <header className="masthead">
        <Lamp ref={lamp} lit={theme === 'dark'} onToggle={toggleTheme} />
        <div className="folio">
          <span>An NLP laboratory, Semester VII</span>
          <Status health={health} />
        </div>
        <motion.div className="double-rule" initial={{ scaleX: 0 }} animate={{ scaleX: 1 }}
                    transition={{ duration: 1.1, ease: EASE }} />
        <motion.h1 className="nameplate" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                   transition={{ delay: 0.45, duration: 0.9, ease: EASE }}>
          NewsMind
        </motion.h1>
        <motion.p className="deck" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9, duration: 0.8 }}>
          Six language models read the news and mark up the copy, the way an editor would: what the story is
          about, which sentences matter, who is named in it, and what it can answer.
          {facts && ` Drawn from ${facts.n_articles.toLocaleString('en')} BBC stories and ${facts.ag_n_test.toLocaleString('en')} AG News headlines.`}
        </motion.p>
      </header>

      <nav className="sections" role="tablist" aria-label="Sections" onKeyDown={onTabKey}>
        {SECTIONS.map((t) => (
          <button key={t.id} id={`tab-${t.id}`} role="tab" aria-selected={tab === t.id}
                  aria-controls={`panel-${t.id}`} tabIndex={tab === t.id ? 0 : -1}
                  className={tab === t.id ? 'section on' : 'section'} onClick={() => setTab(t.id)}>
            {t.label}
            {tab === t.id && (
              <motion.span layoutId="section-rule" className="section-rule"
                           transition={{ type: 'spring', stiffness: 420, damping: 36 }} />
            )}
          </button>
        ))}
      </nav>

      {/* All five sections stay mounted, so results and the interview survive switching;
          only the visible one is shown, with a short CSS entrance. */}
      <main>
        {SECTIONS.map((t) => (
          <div key={t.id} id={`panel-${t.id}`} role="tabpanel" aria-labelledby={`tab-${t.id}`}
               className="view" hidden={tab !== t.id}>
            {panels[t.id]}
          </div>
        ))}
      </main>

      <footer className="colophon">
        <p>
          <em>Colophon.</em> Set in Bodoni Moda and Newsreader. Models: TF-IDF with logistic regression and a
          fine-tuned DistilBERT for topics, TextRank and DistilBART for summaries, spaCy for names, and a
          DistilBERT reader trained on SQuAD for questions. The API is documented at <a href="/docs">/docs</a>.
        </p>
      </footer>
    </div>
  )
}

function Status({ health }) {
  const state = health === null ? 'checking' : health ? 'ok' : 'down'
  const text = {
    checking: 'Connecting to the models…',
    ok: health && `Models loaded, running on ${health.device}`,
    down: 'The API is not running. Start it with uvicorn api.app:app',
  }[state]
  return (
    <span className={`status ${state}`} role="status">
      <motion.span className="status-dot" animate={state === 'checking' ? { opacity: [1, 0.25, 1] } : { opacity: 1 }}
                   transition={state === 'checking' ? { repeat: Infinity, duration: 1.2 } : {}} />
      {text}
    </span>
  )
}
