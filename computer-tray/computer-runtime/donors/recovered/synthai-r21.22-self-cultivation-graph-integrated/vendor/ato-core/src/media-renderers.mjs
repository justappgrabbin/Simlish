import { Automaton } from './automaton.mjs';
export class MediaRendererError extends Error {
    constructor(code, message) { super(message); this.name = 'MediaRendererError'; this.code = code; }
}
export function mediaRendererAutomaton({ id = 'media-renderer', hostRegistry, adapters = {} } = {}) { return new Automaton({ id, address: { mode: 'macro', gate: 22, line: 1, color: 1, tone: 1, base: 1 }, structure: 'trigram', activeLevels: [1, 2], functionalLevel: 'design', ports: [{ id: 'composition', direction: 'input', type: 'media-composition', schemaVersion: '1' }, { id: 'render', direction: 'output', type: 'media-artifact', schemaVersion: '1', guarantees: ['provenance'] }], metadata: { family: 'media-renderer', optionalAdapters: Object.freeze(Object.keys(adapters).sort()) }, implementation: async (input) => { const adapter = adapters[input.renderer]; if (adapter)
        return adapter(input); if (!hostRegistry)
        throw new MediaRendererError('RENDERER_NOT_MOUNTED', input.renderer); const required = input.renderer === 'remotion' ? ['command', 'process'] : input.renderer === 'hyperframes' ? ['browser', 'view'] : ['graphics']; const result = await hostRegistry.execute(required, input); return Object.freeze({ renderer: input.renderer, facultyId: result.facultyId, artifact: result.result, provenance: input.provenance || null }); } }); }
