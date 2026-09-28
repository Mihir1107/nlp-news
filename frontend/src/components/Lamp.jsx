import { forwardRef } from 'react'
import { motion, useReducedMotion } from 'framer-motion'

/**
 * A Victorian gas street lamp, drawn as engraving-style line art. It is the theme switch:
 * lit = night edition (dark theme), out = day edition (light theme).
 * When it is lit, the mantle flickers as it catches, then glows with a slow, uneven breath.
 */
const Lamp = forwardRef(function Lamp({ lit, onToggle }, ref) {
  const reduce = useReducedMotion()
  const catching = reduce ? {} : { opacity: [0, 1, 0.35, 0.9, 0.55, 1] }   // the gas catching
  const breathing = reduce ? {} : { opacity: [1, 0.86, 0.97, 0.9, 1] }      // a living flame, not a bulb

  return (
    <button ref={ref} type="button" className={lit ? 'lamp lit' : 'lamp'} onClick={onToggle}
            aria-pressed={lit} aria-label={lit ? 'Put out the lamp: switch to the day edition (light theme)'
                                               : 'Light the lamp: switch to the night edition (dark theme)'}>
      <svg viewBox="0 0 120 300" aria-hidden="true">
        <defs>
          <radialGradient id="lamp-halo">
            <stop offset="0%" stopColor="var(--lamp)" stopOpacity="0.75" />
            <stop offset="45%" stopColor="var(--lamp)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--lamp)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="lamp-beam" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--lamp)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--lamp)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* light: a halo round the lantern and a beam falling to the pavement */}
        {lit && (
          <motion.g initial={{ opacity: 0 }} animate={catching} transition={{ duration: 0.9, times: [0, 0.15, 0.3, 0.5, 0.7, 1] }}>
            <motion.g animate={breathing} transition={{ duration: 4.5, repeat: Infinity, ease: 'easeInOut', delay: 0.9 }}>
              <circle cx="60" cy="66" r="58" fill="url(#lamp-halo)" />
              <path d="M44 100 L76 100 L112 296 L8 296 Z" fill="url(#lamp-beam)" />
            </motion.g>
          </motion.g>
        )}

        <g className="lamp-lines">
          {/* finial and crown */}
          <circle cx="60" cy="9" r="3.2" className="solid" />
          <path d="M60 12 V22" />
          <path d="M44 38 L52 24 H68 L76 38 Z" className="solid" />
          <path d="M36 40 H84" />
          {/* lantern: tapered glass box with mullions */}
          <path d="M38 42 H82 L75 92 H45 Z" className={lit ? 'glass on' : 'glass'} />
          <path d="M53 42 L56 92 M67 42 L64 92" />
          {/* mantle */}
          {lit
            ? <motion.ellipse cx="60" cy="70" rx="5" ry="8" className="mantle" animate={breathing}
                              transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }} />
            : <ellipse cx="60" cy="70" rx="4" ry="7" className="mantle-off" />}
          {/* lantern base and collar */}
          <path d="M45 92 L55 104 H65 L75 92" className="solid" />
          <path d="M55 104 H65 V112 H55 Z" className="solid" />
          {/* ladder rest */}
          <path d="M40 124 H80" />
          <circle cx="40" cy="124" r="2.6" className="solid" />
          <circle cx="80" cy="124" r="2.6" className="solid" />
          {/* post: fluted shaft, a ring, and the pedestal */}
          <path d="M56.5 112 L55.5 250 H64.5 L63.5 112 Z" className="solid" />
          <path d="M53 178 H67 M53 183 H67" />
          <path d="M52 250 H68 L72 268 H48 Z" className="solid" />
          <path d="M42 268 H78 V276 H42 Z" className="solid" />
          <path d="M8 296 H112" className="kerb" />
        </g>
      </svg>
      <span className="lamp-caption">{lit ? 'Night edition' : 'Day edition'}</span>
    </button>
  )
})

export default Lamp
