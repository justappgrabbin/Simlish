import { normalizeArtifact } from './artifact-normalizer.mjs';

const decoder = new TextDecoder();

const LIMITS = Object.freeze({
  maxZipEntries: 256,
  maxEntryBytes: 4 * 1024 * 1024,
  maxInflatedBytes: 16 * 1024 * 1024,
  maxAtomsPerUnit: 4096,
  maxJsonAtoms: 4096,
});

function unique(values) {
  return [...new Set(values.filter((value) => value !== '' && value != null))];
}

function canonicalAtom(type, value, path = null) {
  return Object.freeze({ type, value, ...(path ? { path } : {}) });
}

function jsStructure(text) {
  const source = String(text ?? '');
  const dependencies = [];
  const exports = [];
  const definitions = [];
  const calls = [];
  const assignments = [];
  const literals = [];
  const operators = [];
  const tokens = [];

  for (const match of source.matchAll(/\bimport\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]/g)) dependencies.push(match[1]);
  for (const match of source.matchAll(/\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)/g)) dependencies.push(match[1]);
  for (const match of source.matchAll(/\bexport\s+(?:default\s+)?(?:class|function|const|let|var)?\s*([A-Za-z_$][\w$]*)?/g)) {
    if (match[1]) exports.push(match[1]);
    else exports.push('default');
  }
  for (const match of source.matchAll(/\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(/g)) definitions.push(match[1]);
  for (const match of source.matchAll(/\bclass\s+([A-Za-z_$][\w$]*)\b/g)) definitions.push(match[1]);
  for (const match of source.matchAll(/\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/g)) assignments.push(match[1]);
  for (const match of source.matchAll(/\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/g)) {
    const call = match[1];
    if (!['if', 'for', 'while', 'switch', 'catch', 'function'].includes(call)) calls.push(call);
  }
  for (const match of source.matchAll(/(?:^|[^\w$])(-?\d+(?:\.\d+)?)\b|(['"`])([^'"`]{0,160})\2|\b(true|false|null|undefined)\b/gm)) {
    const value = match[1] ?? match[3] ?? match[4];
    if (value != null) literals.push(value);
  }
  for (const match of source.matchAll(/===|!==|=>|==|!=|<=|>=|\+\+|--|&&|\|\||\?\?|[+\-*/%<>=!?&|^~:]/g)) operators.push(match[0]);
  for (const match of source.matchAll(/[A-Za-z_$][\w$]*|-?\d+(?:\.\d+)?|===|!==|=>|==|!=|<=|>=|&&|\|\||\?\?|[+\-*/%<>=!?&|^~:]/g)) {
    tokens.push(match[0]);
    if (tokens.length >= LIMITS.maxAtomsPerUnit) break;
  }

  return Object.freeze({
    dependencies: Object.freeze(unique(dependencies)),
    exports: Object.freeze(unique(exports)),
    definitions: Object.freeze(unique(definitions)),
    calls: Object.freeze(unique(calls)),
    assignments: Object.freeze(unique(assignments)),
    literals: Object.freeze(unique(literals).slice(0, 1024)),
    operators: Object.freeze(unique(operators)),
    tokens: Object.freeze(unique(tokens).slice(0, LIMITS.maxAtomsPerUnit)),
    moduleSyntax: /\b(?:import|export)\b/.test(source),
  });
}

function flattenJson(value, path = '$', out = []) {
  if (out.length >= LIMITS.maxJsonAtoms) return out;
  if (value === null || ['string', 'number', 'boolean'].includes(typeof value)) {
    out.push(canonicalAtom('json-value', value, path));
    return out;
  }
  if (Array.isArray(value)) {
    out.push(canonicalAtom('json-array-length', value.length, path));
    value.forEach((member, index) => flattenJson(member, `${path}[${index}]`, out));
    return out;
  }
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort();
    out.push(canonicalAtom('json-object-keys', keys, path));
    for (const key of keys) flattenJson(value[key], `${path}.${key}`, out);
  }
  return out;
}

function binaryAtoms(bytes) {
  const histogram = new Array(256).fill(0);
  for (const value of bytes) histogram[value] += 1;
  const observed = [];
  for (let index = 0; index < histogram.length; index++) {
    if (histogram[index]) observed.push(canonicalAtom('byte-frequency', [index, histogram[index]]));
  }
  const magic = [...bytes.subarray(0, Math.min(bytes.length, 16))];
  return Object.freeze([
    canonicalAtom('byte-length', bytes.length),
    canonicalAtom('magic-bytes', magic),
    ...observed,
  ]);
}

async function inflateRaw(bytes) {
  if (typeof DecompressionStream === 'undefined') return null;
  try {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    return null;
  }
}

function findEndOfCentralDirectory(view) {
  const minimum = Math.max(0, view.byteLength - 65557);
  for (let offset = view.byteLength - 22; offset >= minimum; offset--) {
    if (view.getUint32(offset, true) === 0x06054b50) return offset;
  }
  return -1;
}

async function unpackZip(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = findEndOfCentralDirectory(view);
  if (eocd < 0) return { entries: [], warnings: ['zip-central-directory-not-found'] };

  const entryCount = Math.min(view.getUint16(eocd + 10, true), LIMITS.maxZipEntries);
  const centralOffset = view.getUint32(eocd + 16, true);
  let offset = centralOffset;
  let inflatedTotal = 0;
  const entries = [];
  const warnings = [];

  for (let index = 0; index < entryCount && offset + 46 <= view.byteLength; index++) {
    if (view.getUint32(offset, true) !== 0x02014b50) {
      warnings.push(`zip-central-entry-${index}-invalid`);
      break;
    }
    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const uncompressedSize = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localOffset = view.getUint32(offset + 42, true);
    const nameBytes = bytes.subarray(offset + 46, offset + 46 + nameLength);
    const name = decoder.decode(nameBytes);
    offset += 46 + nameLength + extraLength + commentLength;

    if (!name || name.endsWith('/')) {
      entries.push({ name, directory: true, method, bytes: null, compressedSize, uncompressedSize });
      continue;
    }
    if (uncompressedSize > LIMITS.maxEntryBytes || inflatedTotal + uncompressedSize > LIMITS.maxInflatedBytes) {
      warnings.push(`zip-entry-skipped-size:${name}`);
      entries.push({ name, directory: false, method, bytes: null, compressedSize, uncompressedSize, skipped: 'size-limit' });
      continue;
    }
    if (localOffset + 30 > view.byteLength || view.getUint32(localOffset, true) !== 0x04034b50) {
      warnings.push(`zip-entry-local-header-invalid:${name}`);
      continue;
    }
    const localNameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.subarray(dataStart, dataStart + compressedSize);
    let unpacked = null;
    if (method === 0) unpacked = new Uint8Array(compressed);
    else if (method === 8) unpacked = await inflateRaw(compressed);
    else warnings.push(`zip-entry-unsupported-compression:${name}:${method}`);

    if (unpacked) inflatedTotal += unpacked.byteLength;
    entries.push({ name, directory: false, method, bytes: unpacked, compressedSize, uncompressedSize });
  }
  return { entries, warnings };
}

async function reduceNormalized(normalized, { depth = 0 } = {}) {
  const base = {
    name: normalized.name,
    kind: normalized.kind,
    mime: normalized.mime,
    size: normalized.size,
  };

  if (normalized.kind === 'javascript') {
    const structure = jsStructure(normalized.text);
    return Object.freeze({
      ...base,
      structure,
      primitives: Object.freeze([
        canonicalAtom('artifact-kind', 'javascript'),
        ...structure.tokens.map((value) => canonicalAtom('js-token', value)),
        ...structure.literals.map((value) => canonicalAtom('js-literal', value)),
        ...structure.operators.map((value) => canonicalAtom('js-operator', value)),
      ].slice(0, LIMITS.maxAtomsPerUnit)),
    });
  }

  if (normalized.kind === 'json') {
    try {
      const parsed = JSON.parse(normalized.text);
      return Object.freeze({
        ...base,
        parsed,
        structure: Object.freeze({ dependencies: Object.freeze([]), moduleSyntax: false }),
        primitives: Object.freeze([canonicalAtom('artifact-kind', 'json'), ...flattenJson(parsed)]),
      });
    } catch (error) {
      return Object.freeze({
        ...base,
        structure: Object.freeze({ dependencies: Object.freeze([]), moduleSyntax: false }),
        primitives: Object.freeze([canonicalAtom('artifact-kind', 'json'), canonicalAtom('parse-error', String(error.message))]),
      });
    }
  }

  if (normalized.kind === 'zip') {
    const archive = await unpackZip(normalized.bytes);
    const units = [];
    if (depth < 2) {
      for (const entry of archive.entries) {
        if (!entry.bytes || entry.directory) continue;
        const child = await normalizeArtifact({ name: entry.name, bytes: entry.bytes });
        units.push(await reduceNormalized(child, { depth: depth + 1 }));
      }
    }
    return Object.freeze({
      ...base,
      archive: Object.freeze({
        entries: Object.freeze(archive.entries.map(({ bytes, ...entry }) => Object.freeze(entry))),
        warnings: Object.freeze(archive.warnings),
      }),
      units: Object.freeze(units),
      structure: Object.freeze({
        dependencies: Object.freeze([]),
        moduleSyntax: false,
        archiveEntries: archive.entries.length,
      }),
      primitives: Object.freeze([
        canonicalAtom('artifact-kind', 'zip'),
        canonicalAtom('archive-entry-count', archive.entries.length),
        ...archive.entries.slice(0, LIMITS.maxZipEntries).map((entry) => canonicalAtom('archive-entry', entry.name)),
      ]),
    });
  }

  if (normalized.kind === 'text') {
    const words = unique((normalized.text.match(/[A-Za-z0-9_$.-]+/g) ?? []).slice(0, LIMITS.maxAtomsPerUnit));
    return Object.freeze({
      ...base,
      structure: Object.freeze({ dependencies: Object.freeze([]), moduleSyntax: false }),
      primitives: Object.freeze([
        canonicalAtom('artifact-kind', 'text'),
        ...words.map((word) => canonicalAtom('text-token', word)),
      ]),
    });
  }

  return Object.freeze({
    ...base,
    structure: Object.freeze({ dependencies: Object.freeze([]), moduleSyntax: false }),
    primitives: Object.freeze([canonicalAtom('artifact-kind', normalized.kind), ...binaryAtoms(normalized.bytes)]),
  });
}

export class PrimitiveReducer {
  async reduce(input) {
    const normalized = input?.kind && input?.bytes instanceof Uint8Array ? input : await normalizeArtifact(input);
    const root = await reduceNormalized(normalized);
    const units = root.kind === 'zip' ? root.units : Object.freeze([root]);
    const primitiveCount = units.reduce((sum, unit) => sum + (unit.primitives?.length ?? 0), root.kind === 'zip' ? root.primitives.length : 0);
    return Object.freeze({
      schema: 'synthia.primitive-decomposition.v1',
      artifact: Object.freeze({ name: normalized.name, kind: normalized.kind, mime: normalized.mime, size: normalized.size }),
      root,
      units,
      primitiveCount,
    });
  }
}

export { LIMITS as PRIMITIVE_REDUCTION_LIMITS };
export default PrimitiveReducer;
