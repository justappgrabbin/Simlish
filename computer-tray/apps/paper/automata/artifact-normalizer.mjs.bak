const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

function isBlobLike(value) {
  return value && typeof value === 'object' && typeof value.arrayBuffer === 'function';
}

function extension(name = '') {
  const match = String(name).toLowerCase().match(/\.([a-z0-9_-]+)$/);
  return match?.[1] ?? '';
}

function looksUtf8(bytes) {
  if (!bytes?.length) return true;
  const sample = bytes.subarray(0, Math.min(bytes.length, 4096));
  let controls = 0;
  for (const value of sample) {
    if (value === 0) return false;
    if (value < 9 || (value > 13 && value < 32)) controls += 1;
  }
  return controls / sample.length < 0.03;
}

export function detectArtifactKind({ name = '', mime = '', bytes = new Uint8Array(), text = '' } = {}) {
  const ext = extension(name);
  if (bytes.length >= 4 && bytes[0] === 0x00 && bytes[1] === 0x61 && bytes[2] === 0x73 && bytes[3] === 0x6d) return 'wasm';
  if (bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b && [0x03, 0x05, 0x07].includes(bytes[2]) && [0x04, 0x06, 0x08].includes(bytes[3])) return 'zip';
  if (ext === 'wasm' || mime.includes('wasm')) return 'wasm';
  if (['zip', 'jar', 'apk'].includes(ext) || mime.includes('zip')) return 'zip';
  if (['js', 'mjs', 'cjs', 'jsx', 'ts', 'tsx'].includes(ext) || /javascript|typescript/.test(mime)) return 'javascript';
  if (ext === 'json' || mime.includes('json')) return 'json';
  const trimmed = String(text ?? '').trim();
  if (trimmed && (trimmed.startsWith('{') || trimmed.startsWith('['))) {
    try { JSON.parse(trimmed); return 'json'; } catch {}
  }
  if (text || looksUtf8(bytes)) return 'text';
  return 'binary';
}

export async function normalizeArtifact(input, { name = null, mime = null } = {}) {
  let original = input;
  let resolvedName = name ?? input?.name ?? input?.originalName ?? input?.filename ?? 'artifact';
  let resolvedMime = mime ?? input?.type ?? input?.mime ?? '';
  let bytes;
  let text = '';

  if (typeof input === 'string') {
    text = input;
    bytes = textEncoder.encode(input);
  } else if (input instanceof Uint8Array) {
    bytes = input;
  } else if (input instanceof ArrayBuffer) {
    bytes = new Uint8Array(input);
  } else if (isBlobLike(input)) {
    bytes = new Uint8Array(await input.arrayBuffer());
  } else if (input?.bytes instanceof Uint8Array) {
    bytes = input.bytes;
    text = typeof input.content === 'string' ? input.content : typeof input.originalContent === 'string' ? input.originalContent : '';
  } else if (input?.bytes instanceof ArrayBuffer) {
    bytes = new Uint8Array(input.bytes);
    text = typeof input.content === 'string' ? input.content : '';
  } else if (typeof input?.content === 'string') {
    text = input.content;
    bytes = textEncoder.encode(text);
  } else if (input && typeof input === 'object') {
    text = JSON.stringify(input);
    bytes = textEncoder.encode(text);
    if (!resolvedMime) resolvedMime = 'application/json';
  } else {
    text = String(input ?? '');
    bytes = textEncoder.encode(text);
  }

  if (!text && looksUtf8(bytes)) {
    try { text = textDecoder.decode(bytes); } catch {}
  }

  const kind = detectArtifactKind({ name: resolvedName, mime: resolvedMime, bytes, text });
  return Object.freeze({
    name: String(resolvedName),
    mime: String(resolvedMime || 'application/octet-stream'),
    kind,
    size: bytes.byteLength,
    bytes,
    text,
    original,
  });
}

export function textBytes(text = '') {
  return textEncoder.encode(String(text));
}
