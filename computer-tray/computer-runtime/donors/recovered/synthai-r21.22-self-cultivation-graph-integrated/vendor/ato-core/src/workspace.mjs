const canonicalStringify = value => value === null || typeof value !== 'object' ? JSON.stringify(value) : Array.isArray(value) ? `[${value.map(canonicalStringify).join(',')}]` : `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalStringify(value[key])}`).join(',')}}`;
const stableId = value => { const text = canonicalStringify(value); let a = 2166136261, b = 2246822507; for (const char of text) {
    a = Math.imul(a ^ char.charCodeAt(0), 16777619);
    b = Math.imul(b ^ char.charCodeAt(0), 3266489909);
} return `${(a >>> 0).toString(16).padStart(8, '0')}${(b >>> 0).toString(16).padStart(8, '0')}`; };
export class WorkspaceError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'WorkspaceError'; this.code = code; this.details = details; }
}
const clone = value => structuredClone(value);
const freezeView = view => Object.freeze({ id: view.id, type: view.type, address: view.address || null, automatonId: view.automatonId || null, region: view.region || 'main', order: Number.isFinite(view.order) ? view.order : 0, state: Object.freeze({ ...view.state }) });
export function workspaceSnapshot({ id = 'workspace', address = null, views = [], focus = null, policy = {} } = {}) {
    const snapshot = { id, address, views: views.map(freezeView).sort((a, b) => a.region.localeCompare(b.region) || a.order - b.order || a.id.localeCompare(b.id)), focus, policy: Object.freeze({ ...policy }) };
    return Object.freeze({ ...snapshot, snapshotId: `ws-${stableId(snapshot)}` });
}
export class MorphingWorkspace {
    constructor(initial = {}) { this.current = workspaceSnapshot(initial); this.history = [this.current]; this.future = []; this.transitions = []; this.pins = new Map(); this.branches = new Map(); }
    morph(operations, { operator = 'workspace.patch', rationale = null, automatons = [], policy = null } = {}) {
        if (!Array.isArray(operations) || !operations.length)
            throw new WorkspaceError('EMPTY_MORPH', 'Morph requires operations');
        const source = this.current, next = clone(source);
        delete next.snapshotId;
        next.views = next.views.map(view => ({ ...view, state: { ...view.state } }));
        for (const op of operations)
            this.#apply(next, op);
        if (policy)
            next.policy = { ...next.policy, ...policy };
        const target = workspaceSnapshot(next);
        const transition = Object.freeze({ sequence: this.transitions.length + 1, source: source.snapshotId, target: target.snapshotId, operator, rationale, automatons: Object.freeze([...automatons]), operations: Object.freeze(clone(operations)), inverse: Object.freeze(this.#inverse(source, target)) });
        this.transitions.push(transition);
        this.history.push(target);
        this.current = target;
        this.future = [];
        return transition;
    }
    #apply(ws, op) {
        if (op.type === 'open') {
            if (ws.views.some(view => view.id === op.view.id))
                throw new WorkspaceError('DUPLICATE_VIEW', op.view.id);
            ws.views.push(op.view);
        }
        else if (op.type === 'close') {
            ws.views = ws.views.filter(view => view.id !== op.id);
            if (ws.focus === op.id)
                ws.focus = null;
        }
        else if (op.type === 'replace') {
            const index = ws.views.findIndex(view => view.id === op.id);
            if (index < 0)
                throw new WorkspaceError('VIEW_NOT_FOUND', op.id);
            ws.views[index] = op.view;
        }
        else if (op.type === 'move') {
            const view = ws.views.find(view => view.id === op.id);
            if (!view)
                throw new WorkspaceError('VIEW_NOT_FOUND', op.id);
            view.region = op.region ?? view.region;
            view.order = op.order ?? view.order;
        }
        else if (op.type === 'focus') {
            if (op.id !== null && !ws.views.some(view => view.id === op.id))
                throw new WorkspaceError('VIEW_NOT_FOUND', op.id);
            ws.focus = op.id;
        }
        else if (op.type === 'state') {
            const view = ws.views.find(view => view.id === op.id);
            if (!view)
                throw new WorkspaceError('VIEW_NOT_FOUND', op.id);
            view.state = { ...view.state, ...op.patch };
        }
        else
            throw new WorkspaceError('UNKNOWN_MORPH_OPERATION', op.type);
    }
    #inverse(source, target) { return [{ type: 'restore', snapshot: source.snapshotId, from: target.snapshotId }]; }
    undo() { if (this.history.length < 2)
        return null; const current = this.history.pop(); this.future.push(current); this.current = this.history.at(-1); return this.current; }
    redo() { if (!this.future.length)
        return null; const next = this.future.pop(); this.history.push(next); this.current = next; return next; }
    pin(name) { if (!name)
        throw new WorkspaceError('MISSING_PIN', 'Pin requires a name'); this.pins.set(name, this.current); return this.current; }
    restore(name) { const snapshot = this.pins.get(name); if (!snapshot)
        throw new WorkspaceError('PIN_NOT_FOUND', name); const source = this.current; this.current = snapshot; this.history.push(snapshot); this.future = []; this.transitions.push(Object.freeze({ sequence: this.transitions.length + 1, source: source.snapshotId, target: snapshot.snapshotId, operator: 'workspace.restore', rationale: `restore:${name}`, automatons: Object.freeze([]), operations: Object.freeze([]), inverse: Object.freeze([{ type: 'restore', snapshot: source.snapshotId }]) })); return snapshot; }
    branch(name) { if (!name)
        throw new WorkspaceError('MISSING_BRANCH', 'Branch requires a name'); this.branches.set(name, Object.freeze({ snapshot: this.current, transitionCount: this.transitions.length })); return this.branches.get(name); }
    compare(left, right) { const a = typeof left === 'string' ? (this.pins.get(left) || this.branches.get(left)?.snapshot) : left; const b = typeof right === 'string' ? (this.pins.get(right) || this.branches.get(right)?.snapshot) : right; if (!a || !b)
        throw new WorkspaceError('SNAPSHOT_NOT_FOUND', 'Both snapshots are required'); return Object.freeze({ same: canonicalStringify(a) === canonicalStringify(b), left: a.snapshotId, right: b.snapshotId, opened: Object.freeze(b.views.filter(v => !a.views.some(x => x.id === v.id)).map(v => v.id)), closed: Object.freeze(a.views.filter(v => !b.views.some(x => x.id === v.id)).map(v => v.id)), changed: Object.freeze(b.views.filter(v => a.views.some(x => x.id === v.id && canonicalStringify(x) !== canonicalStringify(v))).map(v => v.id)) }); }
    replay() { return Object.freeze([...this.transitions]); }
}
