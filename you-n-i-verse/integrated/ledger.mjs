// Thin adapter: append-only history landed at the resolved address (spec §18: "resolve address -> land history/bookkeeping").
export class HistoryLedger{
 #by=new Map();#n=0;
 constructor(store=null){this.store=store}
 append(addressKey,entry){const rec=Object.freeze({seq:++this.#n,addressKey,...entry});if(!this.#by.has(addressKey))this.#by.set(addressKey,[]);this.#by.get(addressKey).push(rec);this.store?.append?.('history',rec);return rec}
 recall(addressKey){return [...(this.#by.get(addressKey)||[])]}
 get size(){return this.#n}
 addresses(){return [...this.#by.keys()]}
}

