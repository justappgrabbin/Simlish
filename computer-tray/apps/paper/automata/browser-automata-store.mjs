function clone(value) {
  return value == null ? value : structuredClone(value);
}

export class BrowserAutomataStore {
  constructor({ databaseName = 'synthia-automata-kernel', storeName = 'records', version = 1 } = {}) {
    this.databaseName = databaseName;
    this.storeName = storeName;
    this.version = version;
    this.db = null;
    this.memory = new Map();
    this.mode = 'memory';
  }

  async open() {
    if (this.db || this.mode === 'indexeddb') return this;
    if (typeof indexedDB === 'undefined') {
      this.mode = 'memory';
      return this;
    }
    this.db = await new Promise((resolve, reject) => {
      const request = indexedDB.open(this.databaseName, this.version);
      request.onerror = () => reject(request.error);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(this.storeName)) request.result.createObjectStore(this.storeName);
      };
      request.onsuccess = () => resolve(request.result);
    });
    this.mode = 'indexeddb';
    return this;
  }

  async get(key) {
    await this.open();
    if (this.mode === 'memory') return clone(this.memory.get(String(key)) ?? null);
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, 'readonly');
      const request = tx.objectStore(this.storeName).get(String(key));
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(clone(request.result ?? null));
    });
  }

  async put(key, value) {
    await this.open();
    if (this.mode === 'memory') {
      this.memory.set(String(key), clone(value));
      return true;
    }
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, 'readwrite');
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
      tx.objectStore(this.storeName).put(clone(value), String(key));
    });
  }

  async delete(key) {
    await this.open();
    if (this.mode === 'memory') return this.memory.delete(String(key));
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, 'readwrite');
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
      tx.objectStore(this.storeName).delete(String(key));
    });
  }

  async audit() {
    await this.open();
    return Object.freeze({
      configured: true,
      mode: this.mode,
      durable: this.mode === 'indexeddb',
      databaseName: this.databaseName,
      storeName: this.storeName,
      workerRequired: false,
      backendRequired: false,
    });
  }
}

export default BrowserAutomataStore;
