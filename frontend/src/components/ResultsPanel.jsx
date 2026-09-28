import { motion } from 'framer-motion'
import { Meter } from './ui.jsx'

/** Headline numbers of every experiment, read live from results/*.json through GET /results. */
export default function ResultsPanel({ results }) {
  if (!results) return <p className="empty">Loading results from the API…</p>
  const cls = results.classification
  // Dot plot: the six accuracies are within a few points, so the axis starts near the lowest value
  // (stated in the caption) instead of at 0, where every dot would sit on top of the others.
  const lo = Math.floor(Math.min(...cls.map((c) => c.accuracy)) * 100 - 1) / 100
  const hi = Math.ceil(Math.max(...cls.map((c) => c.accuracy)) * 100 + 1) / 100
  const x = (v) => `${((v - lo) / (hi - lo)) * 100}%`
  const ticks = Array.from({ length: Math.round((hi - lo) * 100) + 1 }, (_, i) => lo + i / 100)
  const best = cls.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))

  return (
    <div className="results">
      <p className="standfirst wide">
        How well each model did on held-out data it never saw in training. Every figure on this page is read
        from the project’s results files, written by the twelve notebooks.
      </p>
      <section className="figure wide">
        <h2>Topic classification on {results.facts.ag_n_test.toLocaleString('en')} AG News test stories</h2>
        <p className="caption">Accuracy of each model; the axis runs from {pct(lo)} to {pct(hi)} so the gaps are visible.</p>
        <div className="dotplot" role="img" aria-label="Accuracy by model">
          <div className="dot-axis">
            {ticks.map((t) => <span key={t} className="tick" style={{ left: x(t) }}>{pct(t)}</span>)}
          </div>
          {[...cls].sort((a, b) => a.accuracy - b.accuracy).map((c, i) => (
            <div key={c.key} className="dot-row">
              <span className="dot-name">{c.model}<span className="muted small"> {fmtMB(c.size_mb)}</span></span>
              <div className="dot-track">
                {ticks.map((t) => <span key={t} className="grid" style={{ left: x(t) }} />)}
                <motion.span className={`dot ${c.key === best.key ? 'best' : ''}`}
                             initial={{ left: '0%', opacity: 0 }} animate={{ left: x(c.accuracy), opacity: 1 }}
                             transition={{ type: 'spring', stiffness: 90, damping: 16, delay: 0.1 + i * 0.08 }} />
                <motion.span className="dot-value" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                             transition={{ delay: 0.6 + i * 0.08 }} style={{ left: x(c.accuracy) }}>
                  {pct(c.accuracy, 1)}
                </motion.span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="figure">
        <h2>Summaries of {results.facts.bart_checked} BBC articles</h2>
        <p className="caption">ROUGE-2 F1 against the reference summaries, which copy sentences from the article, so extractive methods are favoured.</p>
        <Bars rows={results.summarization.map((r) => [r.system, r['rouge2 F1']])} />
      </section>

      <section className="figure">
        <h2>NewsBot on {results.qa.n_answerable} <span className="nowrap">hand-written</span> questions</h2>
        <p className="caption">Share of questions where the right article was retrieved, and how well the answer matched.</p>
        <Bars rows={[
          ['Right article in top 3', results.qa.retrieval_hit_at_3],
          ['Right article first', results.qa.retrieval_hit_at_1],
          ['Answer word overlap (F1)', results.qa.reader_f1],
          ['Exact answer', results.qa.reader_em],
        ]} />
      </section>

      <section className="figure">
        <h2>Tweet sentiment on {results.facts.tweets_n_test.toLocaleString('en')} test tweets</h2>
        <p className="caption">Macro-recall, the benchmark’s official score. RoBERTa was trained on tweets by the benchmark’s authors, so it is a reference ceiling, not a fair rival.</p>
        <Bars rows={results.tweets.map((r) => [r.model.replace(' (reference, not fair)', ' (reference)'), r['macro-recall'], r.model.includes('reference')])} />
      </section>
    </div>
  )
}

function Bars({ rows }) {
  return (
    <ul className="bars">
      {rows.map(([label, v, reference], i) => (
        <li key={label} className={reference ? 'reference' : ''}>
          <span className="bar-label">{label}</span>
          <Meter value={v} delay={0.15 + i * 0.08} color={reference ? 'var(--rule)' : 'var(--pencil)'} />
          <span className="bar-value">{v.toFixed(3)}</span>
        </li>
      ))}
    </ul>
  )
}

const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`
const fmtMB = (mb) => (mb >= 100 ? `${Math.round(mb)} MB` : `${mb} MB`)
