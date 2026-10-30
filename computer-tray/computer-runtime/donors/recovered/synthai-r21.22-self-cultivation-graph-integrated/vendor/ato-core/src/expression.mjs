import { canonicalAddress, normalizeAddress } from './address-space.mjs';
export const EXPRESSION_MODES = Object.freeze(['rest', 'presence', 'conversation', 'focus', 'exploration', 'reflection', 'creation', 'execution', 'celebration', 'repair']);
export class ExpressionError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'ExpressionError'; this.code = code; this.details = details; }
}
export class ExpressionField {
    constructor({ address, mode = 'rest', qualities = {} } = {}) {
        this.address = normalizeAddress(address, { mode: address?.mode || 'macro', allowUnresolved: true });
        this.mode = mode;
        this.qualities = Object.freeze({ ...qualities });
        this.sequence = 0;
        this.events = [];
        this.listeners = new Set();
    }
    express({ source, mode = this.mode, qualities = {}, content = null, context = null, transition = null } = {}) {
        if (!EXPRESSION_MODES.includes(mode))
            throw new ExpressionError('INVALID_EXPRESSION_MODE', mode);
        const previous = Object.freeze({ mode: this.mode, qualities: this.qualities });
        this.mode = mode;
        this.qualities = Object.freeze({ ...this.qualities, ...qualities });
        const event = Object.freeze({
            sequence: ++this.sequence,
            type: 'expression',
            source: source?.id || source || 'system',
            sourceAddress: source?.addressKey || (source?.address ? canonicalAddress(source.address) : null),
            fieldAddress: canonicalAddress(this.address),
            previous,
            current: Object.freeze({ mode: this.mode, qualities: this.qualities }),
            content,
            context,
            transition,
            reversible: true,
        });
        this.events.push(event);
        for (const listener of this.listeners)
            listener(event);
        return event;
    }
    subscribe(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); }
    rollback(sequence) {
        const index = this.events.findIndex((event) => event.sequence === sequence);
        if (index < 0)
            throw new ExpressionError('EXPRESSION_NOT_FOUND', String(sequence));
        const event = this.events[index];
        this.mode = event.previous.mode;
        this.qualities = event.previous.qualities;
        return Object.freeze({ rolledBack: sequence, mode: this.mode, qualities: this.qualities });
    }
    snapshot() { return Object.freeze({ address: this.address, mode: this.mode, qualities: this.qualities, events: Object.freeze([...this.events]) }); }
}
