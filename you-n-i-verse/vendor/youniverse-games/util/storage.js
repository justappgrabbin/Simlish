import { writeFileSync, renameSync, mkdirSync, rmSync } from 'node:fs';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';

export function safeId(value, label = 'identifier') {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(value)) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

export function atomicWrite(path, bytes) {
  mkdirSync(dirname(path), { recursive: true });
  const temp = `${path}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temp, bytes, { mode: 0o600, flush: true });
    renameSync(temp, path);
  } finally {
    rmSync(temp, { force: true });
  }
}

export function writeJSON(path, value) {
  atomicWrite(path, JSON.stringify(value, null, 2));
}
