import { motion, useReducedMotion } from 'framer-motion'

/**
 * Marker: the models' marks on the copy, drawn like an editor's pencil.
 *   variant="wash"   a tinted band behind the words (a highlighter on newsprint)
 *   variant="pencil" a thin underline stroke (the editor's blue pencil)
 * The colour is a background gradient whose width grows from 0% to 100%, so the mark is drawn left to
 * right. `delay` staggers several marks. With "reduce motion" on, the mark simply appears.
 */
export function Marker({ color = 'var(--wash-pencil)', variant = 'wash', delay = 0, children, className = '', ...rest }) {
  const reduce = useReducedMotion()
  const height = variant === 'pencil' ? '0.09em' : '78%'
  return (
    <motion.mark
      className={`marker ${variant} ${className}`}
      style={{ backgroundImage: `linear-gradient(${color}, ${color})` }}
      initial={{ backgroundSize: reduce ? `100% ${height}` : `0% ${height}` }}
      animate={{ backgroundSize: `100% ${height}` }}
      transition={{ duration: reduce ? 0 : 0.6, delay: reduce ? 0 : delay, ease: [0.65, 0, 0.35, 1] }}
      {...rest}
    >
      {children}
    </motion.mark>
  )
}

/** A choice between a few options, set as text; a pencil rule slides under the chosen one. */
export function Segmented({ id, options, value, onChange, label }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      <span className="segmented-label">{label}</span>
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value}
                className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}>
          {o.label}
          {value === o.value && (
            <motion.span layoutId={`seg-${id}`} className="seg-rule"
                         transition={{ type: 'spring', stiffness: 500, damping: 40 }} />
          )}
        </button>
      ))}
    </div>
  )
}

/** Hairline bar that grows to `value` (0–1) with a spring. */
export function Meter({ value, color = 'var(--ink)', delay = 0, height = 3 }) {
  return (
    <div className="meter" style={{ height }}>
      <motion.div className="meter-fill" style={{ background: color }}
                  initial={{ width: 0 }} animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
                  transition={{ type: 'spring', stiffness: 110, damping: 22, delay }} />
    </div>
  )
}

/** Example inputs taken from the real datasets (GET /samples), set as a list of links. */
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

/** Primary action. Shows progress while a request is running. */
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

//   = non-breaking space, so a number never wraps away from its unit
export const ms = (x) => (x < 10 ? `${x.toFixed(1)} ms` : x < 1000 ? `${Math.round(x)} ms` : `${(x / 1000).toFixed(1)} s`)
