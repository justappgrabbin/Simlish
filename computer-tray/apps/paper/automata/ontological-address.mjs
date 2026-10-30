const encoder = new TextEncoder();

function canonical(value) {
  if (value === undefined) return null;
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(canonical);
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
}

export function stableStringify(value) {
  return JSON.stringify(canonical(value));
}

function fnv1a32(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

export async function digest(value) {
  const byteInput = value instanceof Uint8Array
    ? value
    : value instanceof ArrayBuffer
      ? new Uint8Array(value)
      : null;
  const text = byteInput ? null : (typeof value === 'string' ? value : stableStringify(value));
  if (globalThis.crypto?.subtle) {
    const bytes = await globalThis.crypto.subtle.digest('SHA-256', byteInput ?? encoder.encode(text));
    return `sha256:${[...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, '0')).join('')}`;
  }
  if (byteInput) {
    let encoded = '';
    for (const byte of byteInput) encoded += String.fromCharCode(byte);
    return `fnv1a32:${fnv1a32(encoded)}`;
  }
  return `fnv1a32:${fnv1a32(text)}`;
}

function resultBehavior(execution = {}) {
  const value = execution.result?.returnValue ?? execution.result ?? execution.returnValue ?? null;
  return {
    ok: Boolean(execution.ok),
    kind: execution.kind ?? null,
    path: execution.path ?? execution.engine ?? null,
    unresolvedDependencies: execution.unresolvedDependencies ?? [],
    valueType: value == null ? String(value) : Array.isArray(value) ? 'array' : typeof value,
    valueShape: value && typeof value === 'object' && !Array.isArray(value) ? Object.keys(value).sort().slice(0, 64) : null,
    errorClass: execution.ok ? null : String(execution.error ?? '').split(':')[0].slice(0, 120),
  };
}

export class OntologicalAddressSigner {
  async sign({ fingerprint, fieldMap, execution, artifact }) {
    const fieldSignatures = {};
    for (const field of fieldMap.order ?? Object.keys(fieldMap.fields ?? {})) {
      fieldSignatures[field] = await digest(fieldMap.fields?.[field] ?? []);
    }
    const behavior = resultBehavior(execution);
    const behaviorSignature = await digest(behavior);
    const basis = {
      schema: 'synthia.ontological-address.v1',
      artifact: {
        name: artifact?.name ?? null,
        kind: artifact?.kind ?? execution?.kind ?? null,
        fingerprint,
      },
      fieldSignatures,
      behaviorSignature,
    };
    const integritySignature = await digest(basis);
    const compact = integritySignature.split(':')[1]?.slice(0, 24) ?? integritySignature;
    return Object.freeze({
      ...basis,
      id: `onto:${compact}`,
      behavior,
      integritySignature,
      signatureMeaning: 'deterministic integrity digest; not a private-key identity signature',
    });
  }
}

export default OntologicalAddressSigner;
