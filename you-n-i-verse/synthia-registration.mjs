// Simlish registration receipt store. Additive; preserves existing world/character data.
export const REGISTRY_KEY = 'synthia.simlish.registry.v1';
export function registerSimlish(storage, input) {
  if (!storage || typeof storage.getItem !== 'function' || typeof storage.setItem !== 'function') throw new TypeError('Persistent storage required');
  if (!input || typeof input !== 'object' || !input.id || !input.version) throw new TypeError('id and version required');
  const previous = storage.getItem(REGISTRY_KEY);
  let records = {};
  if (previous !== null) {
    try { records = JSON.parse(previous); } catch { throw new Error('Registry is unreadable; refusing to overwrite'); }
    if (!records || typeof records !== 'object' || Array.isArray(records)) throw new Error('Invalid registry; refusing to overwrite');
  }
  const id = String(input.id);
  const record = {
    ...(records[id] || {}),
    id, version: String(input.version),
    computer: 'gaming',
    source: input.source || null,
    address: input.address || null,
    capabilities: Array.isArray(input.capabilities) ? input.capabilities : [],
    updatedAt: new Date().toISOString()
  };
  records[id] = record;
  storage.setItem(REGISTRY_KEY, JSON.stringify(records));
  return record;
}
export function readSimlishRegistrations(storage) {
  const raw = storage.getItem(REGISTRY_KEY);
  return raw === null ? {} : JSON.parse(raw);
}
