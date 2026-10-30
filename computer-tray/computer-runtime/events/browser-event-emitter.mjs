import { buildComputerEvent } from './event-grammar.mjs';

/**
 * Browser storage for Computer event grammar-v1.
 *
 * IndexedDB events are inserted with add(), never rewritten or erased.
 * The object store's monotonically increasing key determines replay order.
 * Existing ComputerFormationService consumes this exact emit/read/history API.
 * No server or web worker is involved.
 */
export class BrowserEventEmitter {
  constructor({ bus = null, namespace = 'synthai-computer-web', indexedDBImpl = globalThis.indexedDB } = {}) {
    this.bus = bus;
    this.namespace = namespace;
    this.indexedDB = indexedDBImpl;
    if (!this.indexedDB) throw new Error('IndexedDB unavailable: persistent Computer event history cannot be guaranteed');
  }

  buildEvent(partial = {}) { return buildComputerEvent(partial); }

  async #open() {
    const req = this.indexedDB.open('synthai-computer-events-' + this.namespace, 1);
    return new Promise((resolve, reject) => {
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains('events'))
          req.result.createObjectStore('events', { autoIncrement: true });
      };
      req.onerror = () => reject(req.error ?? new Error('Computer event database open failed'));
      req.onblocked = () => reject(new Error('Computer event database blocked by another tab'));
      req.onsuccess = () => resolve(req.result);
    });
  }

  async emitEvent(partial = {}) {
    const event = this.buildEvent(partial);
    const db = await this.#open();
    try {
      await new Promise((resolve, reject) => {
        const tx = db.transaction('events', 'readwrite');
        tx.objectStore('events').add(event);
        tx.oncomplete = () => resolve();
        tx.onabort = tx.onerror = () => reject(tx.error ?? new Error('Computer event append failed'));
      });
    } catch (error) {
      this.bus?.emit('event:persistence-failure', { event_id: event.event_id, error: String(error?.message ?? error) });
      throw error;
    } finally { db.close(); }
    this.bus?.emit('event:emitted', event);
    this.bus?.emit('event:' + event.event_type, event);
    return event;
  }

  async readAll() {
    const db = await this.#open();
    try {
      return await new Promise((resolve, reject) => {
        const events = [];
        const tx = db.transaction('events', 'readonly');
        const request = tx.objectStore('events').openCursor();
        request.onsuccess = () => {
          const cursor = request.result;
          if (cursor) { events.push(cursor.value); cursor.continue(); }
        };
        request.onerror = () => reject(request.error ?? new Error('Computer history read failed'));
        tx.oncomplete = () => resolve(events);
        tx.onerror = tx.onabort = () => reject(tx.error ?? new Error('Computer history transaction failed'));
      });
    } finally { db.close(); }
  }

  async history(trajectory_id) {
    return (await this.readAll()).filter(e => e.trajectory_id === trajectory_id);
  }

  async forEntity(id) {
    return (await this.readAll()).filter(e => e.actor_id === id || e.target_id === id);
  }
}
