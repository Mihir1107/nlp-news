// Small wrapper around fetch that turns API errors into sentences a person can act on.
export async function api(path, body) {
  let res
  try {
    res = await fetch(path, body === undefined ? undefined : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new Error("Can't reach the API. Start it with `uvicorn api.app:app`, then reload this page.")
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    // Pydantic prefixes custom messages with "Value error, "; people do not need to see that.
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg.replace(/^Value error, /, '')).join('; ')
      : data.detail
    if (res.status === 422) throw new Error(`Check the input: ${detail}.`)
    throw new Error(`The API answered ${res.status}${detail ? `: ${detail}` : ''}.`)
  }
  return data
}

// Colours shared by every view: printed spot colours. `ink` for bars and pencil strokes,
// `wash` for the tinted band behind marked words. Values live in styles.css (:root).
export const TOPIC_COLOR = {
  World: { ink: 'var(--ultramarine)', wash: 'var(--wash-ultramarine)' },
  Sports: { ink: 'var(--viridian)', wash: 'var(--wash-viridian)' },
  Business: { ink: 'var(--ochre)', wash: 'var(--wash-ochre)' },
  'Sci/Tech': { ink: 'var(--madder)', wash: 'var(--wash-madder)' },
}

export const ENTITY_KIND = {
  PERSON: ['person', 'var(--wash-madder)'],
  ORG: ['organisation', 'var(--wash-ultramarine)'],
  GPE: ['place', 'var(--wash-viridian)'], LOC: ['place', 'var(--wash-viridian)'], FAC: ['place', 'var(--wash-viridian)'],
  NORP: ['group', 'var(--wash-slate)'],
  DATE: ['date or time', 'var(--wash-ochre)'], TIME: ['date or time', 'var(--wash-ochre)'],
  MONEY: ['number', 'var(--wash-plum)'], PERCENT: ['number', 'var(--wash-plum)'], CARDINAL: ['number', 'var(--wash-plum)'],
  QUANTITY: ['number', 'var(--wash-plum)'], ORDINAL: ['number', 'var(--wash-plum)'],
}
export const entityKind = (label) => ENTITY_KIND[label] || ['other', 'var(--wash-slate)']
