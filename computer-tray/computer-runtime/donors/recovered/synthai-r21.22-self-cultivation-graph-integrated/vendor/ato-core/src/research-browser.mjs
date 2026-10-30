import { Automaton } from './automaton.mjs';
export class ResearchError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'ResearchError'; this.code = code; this.details = details; }
}
export class ResearchWorkspace {
    constructor() { this.projects = new Map(); this.sequence = 0; }
    create({ id, title, question, scope = null } = {}) { const project = { id: id || `research-${++this.sequence}`, title, question, scope, sources: new Map(), claims: new Map(), notes: [], hypotheses: new Map(), experiments: new Map(), drafts: [], status: 'researching' }; this.projects.set(project.id, project); return this.snapshot(project.id); }
    source(projectId, { id, url, title, excerpt = null, author = null, published = null, capturedBy = 'browser' } = {}) { const p = this.#project(projectId), record = Object.freeze({ id: id || `source-${++this.sequence}`, url, title, excerpt, author, published, capturedBy, kind: 'source' }); p.sources.set(record.id, record); return record; }
    claim(projectId, { id, text, sourceIds = [], kind = 'source-claim', confidence = null } = {}) { const p = this.#project(projectId); for (const sourceId of sourceIds)
        if (!p.sources.has(sourceId))
            throw new ResearchError('SOURCE_NOT_FOUND', sourceId); const record = Object.freeze({ id: id || `claim-${++this.sequence}`, text, sourceIds: Object.freeze([...sourceIds]), kind, confidence }); p.claims.set(record.id, record); return record; }
    note(projectId, { text, sourceIds = [], claimIds = [] } = {}) { const p = this.#project(projectId), record = Object.freeze({ id: `note-${++this.sequence}`, text, sourceIds: Object.freeze([...sourceIds]), claimIds: Object.freeze([...claimIds]) }); p.notes.push(record); return record; }
    hypothesis(projectId, { id, statement, evidenceClaimIds = [] } = {}) { const p = this.#project(projectId), record = Object.freeze({ id: id || `hypothesis-${++this.sequence}`, statement, evidenceClaimIds: Object.freeze([...evidenceClaimIds]), status: 'proposed' }); p.hypotheses.set(record.id, record); return record; }
    experiment(projectId, { id, hypothesisId, protocol, executor = 'local', effects = [] } = {}) { const p = this.#project(projectId); if (!p.hypotheses.has(hypothesisId))
        throw new ResearchError('HYPOTHESIS_NOT_FOUND', hypothesisId); const record = { id: id || `experiment-${++this.sequence}`, hypothesisId, protocol, executor, effects: Object.freeze([...effects]), runs: [], status: 'designed' }; p.experiments.set(record.id, record); return this.#experimentView(record); }
    async run(projectId, experimentId, { hostRegistry, input, authorization = {} } = {}) { const p = this.#project(projectId), experiment = p.experiments.get(experimentId); if (!experiment)
        throw new ResearchError('EXPERIMENT_NOT_FOUND', experimentId); const consequential = experiment.effects.filter(effect => !['read', 'local-compute', 'local-write'].includes(effect)); if (consequential.some(effect => !authorization[effect]))
        throw new ResearchError('EXPERIMENT_AUTHORIZATION_REQUIRED', 'Consequential experiment effects need authorization', { effects: consequential }); if (!hostRegistry)
        throw new ResearchError('HOST_REQUIRED', 'Experiment needs mounted host'); const result = await hostRegistry.execute(experiment.protocol.requiredCapabilities || ['execute-code'], input, { projectId, experimentId }); const run = Object.freeze({ id: `run-${++this.sequence}`, facultyId: result.facultyId, input, result: result.result }); experiment.runs.push(run); experiment.status = 'observed'; return run; }
    draft(projectId, { title, sections = [] } = {}) { const p = this.#project(projectId); const unresolved = []; const rendered = sections.map(section => { const claims = (section.claimIds || []).map(id => { const claim = p.claims.get(id); if (!claim) {
        unresolved.push(id);
        return `[UNRESOLVED CLAIM ${id}]`;
    } const cites = claim.sourceIds.map(sourceId => `[${sourceId}]`).join(''); return `${claim.text}${cites}`; }); return Object.freeze({ heading: section.heading, content: Object.freeze(claims) }); }); const bibliography = Object.freeze([...p.sources.values()].map(source => Object.freeze({ id: source.id, title: source.title, url: source.url, author: source.author, published: source.published }))); const draft = Object.freeze({ id: `draft-${++this.sequence}`, title, sections: Object.freeze(rendered), bibliography, unresolved: Object.freeze(unresolved), kind: 'research-draft' }); p.drafts.push(draft); return draft; }
    snapshot(projectId) { const p = this.#project(projectId); return Object.freeze({ id: p.id, title: p.title, question: p.question, scope: p.scope, status: p.status, sources: Object.freeze([...p.sources.values()]), claims: Object.freeze([...p.claims.values()]), notes: Object.freeze([...p.notes]), hypotheses: Object.freeze([...p.hypotheses.values()]), experiments: Object.freeze([...p.experiments.values()].map(x => this.#experimentView(x))), drafts: Object.freeze([...p.drafts]) }); }
    #experimentView(x) { return Object.freeze({ id: x.id, hypothesisId: x.hypothesisId, protocol: x.protocol, executor: x.executor, effects: x.effects, runs: Object.freeze([...x.runs]), status: x.status }); }
    #project(id) { const p = this.projects.get(id); if (!p)
        throw new ResearchError('PROJECT_NOT_FOUND', id); return p; }
}
export function researchBrowserAutomaton({ workspace = new ResearchWorkspace() } = {}) { return new Automaton({ id: 'research-browser', address: { mode: 'macro', gate: 11, line: 1, color: 1, tone: 1, base: 1 }, structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5], functionalLevel: 'mind', ports: [{ id: 'research-request', direction: 'input', type: 'research', schemaVersion: '1' }, { id: 'research-record', direction: 'output', type: 'research', schemaVersion: '1', guarantees: ['provenance'] }], state: workspace, metadata: { family: 'research-browser', selfReferential: true, faculties: ['research', 'citations', 'experiments', 'papers'] }, implementation: (input, { state }) => { const op = input.operation; if (op === 'create')
        return state.create(input.project); if (op === 'source')
        return state.source(input.projectId, input.source); if (op === 'claim')
        return state.claim(input.projectId, input.claim); if (op === 'note')
        return state.note(input.projectId, input.note); if (op === 'hypothesis')
        return state.hypothesis(input.projectId, input.hypothesis); if (op === 'experiment')
        return state.experiment(input.projectId, input.experiment); if (op === 'draft')
        return state.draft(input.projectId, input.draft); if (op === 'snapshot')
        return state.snapshot(input.projectId); throw new ResearchError('UNKNOWN_OPERATION', op); } }); }
