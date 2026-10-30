/**
 * JsonFilePersistence — file-backed implementation of the existing kernel
 * persistence pattern (same {load(key), save(key, value)} interface as
 * MemoryPersistence / LocalStoragePersistence in core/kernel.mjs).
 * Enables restart recovery of StateStore state in Node (tests + local runtime).
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export class JsonFilePersistence {
  constructor(filePath) {
    if (!filePath) throw new Error('JsonFilePersistence: filePath required');
    this.filePath = filePath;
  }

  async load(key) {
    let text;
    try {
      text = await readFile(this.filePath, 'utf8');
    } catch (error) {
      if (error?.code === 'ENOENT') return undefined;
      throw error;
    }
    const db = JSON.parse(text || '{}');
    return db[key];
  }

  async save(key, value) {
    let db = {};
    try {
      db = JSON.parse(await readFile(this.filePath, 'utf8') || '{}');
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
    db[key] = value;
    await mkdir(dirname(this.filePath), { recursive: true });
    await writeFile(this.filePath, JSON.stringify(db), 'utf8');
    return true;
  }
}

export default JsonFilePersistence;
