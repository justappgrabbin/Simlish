// Video queue manager with priority. Lower priority number = sooner.

export class VideoQueue {
  constructor() {
    this._items = [];
    this._seq = 0;
  }

  /** @returns job id */
  enqueue({ sceneDNA, priority = 5 }) {
    if (!sceneDNA?.hash) throw new Error('VideoQueue.enqueue: sceneDNA with hash required');
    const job = { id: `vq-${this._seq++}`, sceneDNA, priority, seq: this._seq, status: 'queued' };
    this._items.push(job);
    this._items.sort((a, b) => a.priority - b.priority || a.seq - b.seq);
    return job.id;
  }

  dequeue() {
    return this._items.shift() ?? null;
  }

  peek() { return this._items[0] ?? null; }
  size() { return this._items.length; }
}
