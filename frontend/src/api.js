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

// Colours shared by every view: one highlighter per topic and per entity family.
export const TOPIC_COLOR = { World: 'var(--sky)', Sports: 'var(--mint)', Business: 'var(--yellow)', 'Sci/Tech': 'var(--pink)' }

export const ENTITY_KIND = {
  PERSON: ['person', 'var(--pink)'],
  ORG: ['organisation', 'var(--sky)'],
  GPE: ['place', 'var(--mint)'], LOC: ['place', 'var(--mint)'], FAC: ['place', 'var(--mint)'],
  NORP: ['group', 'var(--lilac)'],
  DATE: ['date or time', 'var(--yellow)'], TIME: ['date or time', 'var(--yellow)'],
  MONEY: ['number', 'var(--apricot)'], PERCENT: ['number', 'var(--apricot)'], CARDINAL: ['number', 'var(--apricot)'],
  QUANTITY: ['number', 'var(--apricot)'], ORDINAL: ['number', 'var(--apricot)'],
}
export const entityKind = (label) => ENTITY_KIND[label] || ['other', 'var(--lilac)']
