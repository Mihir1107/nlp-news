import { motion, useReducedMotion } from 'framer-motion'

/**
 * Marker: text highlighted as if with a highlighter pen. The colour is a background gradient whose
 * width grows from 0% to 100%, so the mark "sweeps" left to right. `delay` staggers several marks.
 */
export function Marker({ color = 'var(--yellow)', delay = 0, height = '82%', children, className = '', ...rest }) {
  const reduce = useReducedMotion()
  // `height` is how much of the line the ink covers, from the bottom, like a real highlighter stroke.
  return (
    <motion.mark
      className={`marker ${className}`}
      style={{ backgroundImage: `linear-gradient(${color}, ${color})` }}
      initial={{ backgroundSize: reduce ? `100% ${height}` : `0% ${height}` }}
      animate={{ backgroundSize: `100% ${height}` }}
      transition={{ duration: reduce ? 0 : 0.55, delay: reduce ? 0 : delay, ease: [0.65, 0, 0.35, 1] }}
      {...rest}
    >
      {children}
    </motion.mark>
  )
}

/** Segmented control. The selected pill slides between options (shared layoutId). */
export function Segmented({ id, options, value, onChange, label }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value}
                className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}>
          {value === o.value && (
            <motion.span layoutId={`seg-${id}`} className="seg-pill"
                         transition={{ type: 'spring', stiffness: 500, damping: 38 }} />
          )}
          <span className="seg-text">{o.label}</span>
        </button>
      ))}
    </div>
  )
}

/** Horizontal bar that grows to `value` (0–1) with a spring. */
export function Meter({ value, color = 'var(--ink)', delay = 0, height = 10 }) {
  return (
    <div className="meter" style={{ height }}>
      <motion.div className="meter-fill" style={{ background: color }}
                  initial={{ width: 0 }} animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
                  transition={{ type: 'spring', stiffness: 120, damping: 20, delay }} />
    </div>
  )
}

/** Row of example inputs taken from the real datasets (GET /samples). */
export function Samples({ label, items, onPick }) {
  if (!items?.length) return null
  return (
    <div className="samples">
      <span className="samples-label">{label}</span>
      {items.map((it) => (
        <button key={it.key} type="button" className="chip" onClick={() => onPick(it)} title={it.title}>
          {it.short}
        </button>
      ))}
    </div>
  )
}

export function ErrorNote({ error }) {
  if (!error) return null
  return (
    <motion.p className="error" role="alert" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}>
      {error}
    </motion.p>
  )
}

/** Primary action button that shows progress while a request is running. */
export function RunButton({ busy, children, busyText = 'Working…', ...rest }) {
  return (
    <button type="submit" className="run" disabled={busy} {...rest}>
      <span className={busy ? 'run-label busy' : 'run-label'}>{busy ? busyText : children}</span>
    </button>
  )
}

export const shorten = (text, words = 7) => {
  const w = text.split(/\s+/)
  return w.length <= words ? text : `${w.slice(0, words).join(' ')}…`
}

// \u00a0 = non-breaking space, so a number never wraps away from its unit
export const ms = (x) => (x < 10 ? `${x.toFixed(1)}\u00a0ms` : x < 1000 ? `${Math.round(x)}\u00a0ms` : `${(x / 1000).toFixed(1)}\u00a0s`)
