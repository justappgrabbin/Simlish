const stableId = value => { const text = JSON.stringify(value); let h = 2166136261; for (const char of text)
    h = Math.imul(h ^ char.charCodeAt(0), 16777619); return (h >>> 0).toString(16).padStart(8, '0'); };
export const COMPANION_MODES = Object.freeze(['rest', 'presence', 'conversation', 'task', 'reflection', 'maintenance', 'growth']);
export class CompanionError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'CompanionError'; this.code = code; this.details = details; }
}
export class CompanionAutomata {
    constructor({ id = 'companion', mesh, workspace, address, policy = {} } = {}) {
        if (!mesh || !workspace)
            throw new CompanionError('MISSING_SUBSTRATE', 'Companion requires a mesh and morphing workspace');
        this.id = id;
        this.mesh = mesh;
        this.workspace = workspace;
        this.address = address || null;
        this.mode = 'rest';
        this.policy = Object.freeze({ autonomousTasks: false, autonomousMorphs: false, autonomousGrowth: false, maxActionsPerCycle: 8, ...policy });
        this.events = [];
        this.sequence = 0;
        this.pending = [];
    }
    transition(mode, { reason = null, reversible = true } = {}) { if (!COMPANION_MODES.includes(mode))
        throw new CompanionError('INVALID_MODE', mode); const event = Object.freeze({ sequence: ++this.sequence, type: 'mode', from: this.mode, to: mode, reason, reversible }); this.mode = mode; this.events.push(event); return event; }
    introspect() { const view = { id: this.id, mode: this.mode, address: this.address, policy: this.policy, mesh: this.mesh.snapshot(), workspace: this.workspace.current, pending: Object.freeze([...this.pending]), eventCount: this.events.length }; return Object.freeze({ ...view, selfId: `self-${stableId(view)}` }); }
    async talk(input, { startAutomaton = 'conversation', contributions = [] } = {}) { this.transition('conversation', { reason: 'person initiated conversation' }); let output; if (this.mesh.automatons.has(startAutomaton)) {
        output = await this.mesh.automatons.get(startAutomaton).call(input, { contributions, companion: this });
    }
    else
        output = `${String(input).trim()} — I am here.`; this.events.push(Object.freeze({ sequence: ++this.sequence, type: 'utterance', input, output, mode: this.mode })); return output; }
    propose(action) { const proposal = Object.freeze({ id: `proposal-${stableId({ action, sequence: this.sequence })}`, action, status: 'pending', createdAtSequence: this.sequence }); this.pending.push(proposal); this.events.push(Object.freeze({ sequence: ++this.sequence, type: 'proposal', proposal })); return proposal; }
    async act(proposalId, { authorize = false } = {}) { const index = this.pending.findIndex(item => item.id === proposalId); if (index < 0)
        throw new CompanionError('PROPOSAL_NOT_FOUND', proposalId); const proposal = this.pending[index]; const permitted = authorize || (proposal.action.kind === 'task' && this.policy.autonomousTasks) || (proposal.action.kind === 'morph' && this.policy.autonomousMorphs) || (proposal.action.kind === 'growth' && this.policy.autonomousGrowth); if (!permitted)
        return Object.freeze({ status: 'awaiting-authorization', proposal }); this.pending.splice(index, 1); this.transition(proposal.action.kind === 'morph' ? 'task' : proposal.action.kind, { reason: proposal.id }); let result; if (proposal.action.kind === 'task')
        result = await this.mesh.run(proposal.action.start, proposal.action.input, proposal.action.options);
    else if (proposal.action.kind === 'morph')
        result = this.workspace.morph(proposal.action.operations, proposal.action.context);
    else
        result = Object.freeze({ status: 'authorized', action: proposal.action }); this.events.push(Object.freeze({ sequence: ++this.sequence, type: 'action', proposalId, result })); return Object.freeze({ status: 'complete', result }); }
    rest(reason = 'no action needed') { return this.transition('rest', { reason }); }
    replay() { return Object.freeze([...this.events]); }
}
