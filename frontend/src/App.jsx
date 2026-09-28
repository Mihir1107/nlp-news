import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { api } from './api.js'
import { Marker } from './components/ui.jsx'
import ClassifyPanel from './components/ClassifyPanel.jsx'
import SummarizePanel from './components/SummarizePanel.jsx'
import EntitiesPanel from './components/EntitiesPanel.jsx'
import AskPanel from './components/AskPanel.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'

const TABS = [
  { id: 'topic', label: 'Topic' },
  { id: 'summary', label: 'Summary' },
  { id: 'names', label: 'Names and places' },
  { id: 'ask', label: 'Ask the archive' },
  { id: 'results', label: 'How good is it?' },
]

export default function App() {
  const [tab, setTab] = useState('topic')
  const [health, setHealth] = useState(null)     // null = checking, false = unreachable
  const [samples, setSamples] = useState(null)
  const [results, setResults] = useState(null)

  // One round of requests when the page opens: API status, example inputs, evaluation numbers.
  useEffect(() => {
    api('/health').then(setHealth).catch(() => setHealth(false))
    api('/samples').then(setSamples).catch(() => {})
    api('/results').then(setResults).catch(() => {})
  }, [])

  // Arrow keys move between tabs (standard tablist behaviour).
  function onTabKey(e) {
    const i = TABS.findIndex((t) => t.id === tab)
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!step) return
    const next = TABS[(i + step + TABS.length) % TABS.length].id
    setTab(next)
    document.getElementById(`tab-${next}`)?.focus()
  }

  const panels = {
    topic: <ClassifyPanel samples={samples} />,
    summary: <SummarizePanel samples={samples} facts={results?.facts} />,
    names: <EntitiesPanel samples={samples} facts={results?.facts} />,
    ask: <AskPanel samples={samples} facts={results?.facts} />,
    results: <ResultsPanel results={results} />,
  }

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <h1 className="wordmark"><Marker delay={0.35} height="38%">NewsMind</Marker></h1>
          <p className="tagline">Paste a news story and see what each model reads in it.</p>
        </div>
        <Status health={health} />
      </header>

      <nav className="tabs" role="tablist" aria-label="Tools" onKeyDown={onTabKey}>
        {TABS.map((t) => (
          <button key={t.id} id={`tab-${t.id}`} role="tab" aria-selected={tab === t.id} aria-controls={`panel-${t.id}`}
                  tabIndex={tab === t.id ? 0 : -1} className={tab === t.id ? 'tab on' : 'tab'} onClick={() => setTab(t.id)}>
            {tab === t.id && (
              <motion.span layoutId="tab-marker" className="tab-marker"
                           transition={{ type: 'spring', stiffness: 420, damping: 34 }} />
            )}
            <span className="tab-text">{t.label}</span>
          </button>
        ))}
      </nav>

      {/* All five views stay mounted, so results and the chat survive switching tabs;
          only the visible one is shown, with a short CSS entrance animation. */}
      <main>
        {TABS.map((t) => (
          <div key={t.id} id={`panel-${t.id}`} role="tabpanel" aria-labelledby={`tab-${t.id}`}
               className="view" hidden={tab !== t.id}>
            {panels[t.id]}
          </div>
        ))}
      </main>

      <footer className="footer">
        NewsMind, Semester VII NLP lab project. Models: TF-IDF + logistic regression, DistilBERT, TextRank,
        DistilBART, spaCy, DistilBERT-SQuAD. API documentation at <a href="/docs">/docs</a>.
      </footer>
    </div>
  )
}

function Status({ health }) {
  const state = health === null ? 'checking' : health ? 'ok' : 'down'
  const text = {
    checking: 'Connecting to the API…',
    ok: health && `API ready on ${health.device}`,
    down: 'API not reachable. Run uvicorn api.app:app',
  }[state]
  return (
    <p className={`status ${state}`} role="status">
      <motion.span className="status-dot" animate={state === 'checking' ? { opacity: [1, 0.3, 1] } : { opacity: 1 }}
                   transition={state === 'checking' ? { repeat: Infinity, duration: 1.2 } : {}} />
      {text}
    </p>
  )
}
