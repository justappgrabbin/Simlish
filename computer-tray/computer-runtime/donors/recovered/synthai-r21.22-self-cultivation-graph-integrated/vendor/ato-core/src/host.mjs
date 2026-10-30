export class HostError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'HostError'; this.code = code; this.details = details; }
}
export class HostRegistry {
    constructor() { this.faculties = new Map(); this.events = []; this.sequence = 0; }
    register(faculty) { if (!faculty?.id || typeof faculty.execute !== 'function')
        throw new HostError('INVALID_FACULTY', 'Faculty needs id and execute'); this.faculties.set(faculty.id, Object.freeze({ ...faculty, capabilities: Object.freeze([...(faculty.capabilities || [])].sort()) })); return faculty; }
    candidates(required = []) { return [...this.faculties.values()].filter(f => required.every(cap => f.capabilities.includes(cap))).sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100) || a.id.localeCompare(b.id)); }
    resolve(required = []) { const faculty = this.candidates(required)[0]; if (!faculty)
        throw new HostError('LOCAL_FACULTY_NOT_FOUND', 'No local faculty satisfies the requested effects', { required }); return faculty; }
    async execute(required, input, context = {}) { const faculty = this.resolve(required); const result = await faculty.execute(input, Object.freeze({ ...context, facultyId: faculty.id })); this.events.push(Object.freeze({ sequence: ++this.sequence, facultyId: faculty.id, required: Object.freeze([...required]), input, result })); return Object.freeze({ facultyId: faculty.id, result }); }
    replay() { return Object.freeze([...this.events]); }
}
export function createAcodeHost({ terminal, files, views, storage, workers, graphics } = {}) {
    const host = new HostRegistry();
    if (terminal)
        host.register({ id: 'acode.terminal', priority: 10, capabilities: ['command', 'compile', 'encode-media', 'execute-binary', 'execute-code', 'process', 'python', 'shell'], execute: terminal });
    if (workers)
        host.register({ id: 'acode.worker', priority: 5, capabilities: ['execute-code', 'javascript', 'pure', 'worker'], execute: workers });
    host.register({ id: 'acode.javascript', priority: 1, capabilities: ['execute-code', 'javascript', 'pure'], execute: async (input) => typeof input === 'function' ? input() : input });
    if (files)
        host.register({ id: 'acode.files', priority: 5, capabilities: ['file-read', 'file-write', 'project'], execute: files });
    if (views)
        host.register({ id: 'acode.views', priority: 5, capabilities: ['browser', 'interface', 'view'], execute: views });
    if (storage)
        host.register({ id: 'acode.storage', priority: 5, capabilities: ['database', 'persistence', 'storage'], execute: storage });
    if (graphics)
        host.register({ id: 'acode.graphics', priority: 5, capabilities: ['canvas', 'game', 'graphics', 'image', 'video', 'webgl', 'webgpu'], execute: graphics });
    return host;
}
export class LocalRemounter {
    constructor(host) { this.host = host; }
    plan(manifest) {
        const boundaries = manifest.boundaries || [];
        const replacements = boundaries.map(boundary => {
            if (['http-server', 'express', 'websocket-server', 'remote-api'].includes(boundary.kind))
                return Object.freeze({ boundary, replacement: 'local-channel', faculty: 'acode.javascript' });
            if (['supabase', 'remote-db', 'server-storage'].includes(boundary.kind))
                return Object.freeze({ boundary, replacement: 'local-persistence', faculty: 'acode.storage' });
            if (['python-service', 'shell-service', 'media-server'].includes(boundary.kind))
                return Object.freeze({ boundary, replacement: 'terminal-process', faculty: 'acode.terminal' });
            if (['remote-ui', 'server-rendered-ui'].includes(boundary.kind))
                return Object.freeze({ boundary, replacement: 'local-view', faculty: 'acode.views' });
            return Object.freeze({ boundary, replacement: 'preserve-local', faculty: null });
        });
        return Object.freeze({ artifactId: manifest.id, replacements: Object.freeze(replacements), backendRequired: false });
    }
}
