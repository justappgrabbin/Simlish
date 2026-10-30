export class AutonomyError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'AutonomyError'; this.code = code; this.details = details; }
}
const clamp = value => Math.max(0, Math.min(1, value));
export class ConfidenceLedger {
    constructor() { this.classes = new Map(); this.policies = new Map(); this.events = []; this.sequence = 0; }
    grant(taskClass, { threshold = 0.8, destinations = [], dataScopes = [], effects = [], expiresAt = null } = {}) { if (!taskClass)
        throw new AutonomyError('MISSING_TASK_CLASS', 'Task class required'); const policy = Object.freeze({ taskClass, threshold: clamp(threshold), destinations: Object.freeze([...destinations].sort()), dataScopes: Object.freeze([...dataScopes].sort()), effects: Object.freeze([...effects].sort()), expiresAt }); this.policies.set(taskClass, policy); this.#event('grant', policy); return policy; }
    revoke(taskClass) { const policy = this.policies.get(taskClass); this.policies.delete(taskClass); this.#event('revoke', { taskClass }); return Boolean(policy); }
    record(taskClass, { kind, weight = 1, context = null } = {}) { const valid = ['confirmed', 'success', 'correction', 'reversal', 'denial', 'failure', 'context-change']; if (!valid.includes(kind))
        throw new AutonomyError('INVALID_EVIDENCE', kind); const row = this.classes.get(taskClass) || { positive: 0, negative: 0, total: 0, evidence: [] }; const effect = { confirmed: 1, success: 1, correction: -1, reversal: -1, denial: -1.5, failure: -1.5, 'context-change': -0.5 }[kind] * Math.max(0, weight); if (effect >= 0)
        row.positive += effect;
    else
        row.negative += Math.abs(effect); row.total += Math.abs(effect); row.evidence.push(Object.freeze({ sequence: ++this.sequence, kind, weight, context })); this.classes.set(taskClass, row); return this.score(taskClass); }
    score(taskClass) { const row = this.classes.get(taskClass) || { positive: 0, negative: 0, total: 0, evidence: [] }; const priorSuccess = 1, priorTotal = 2; const score = (row.positive + priorSuccess) / (row.positive + row.negative + priorTotal); return Object.freeze({ taskClass, score: clamp(score), positive: row.positive, negative: row.negative, evidenceCount: row.evidence.length, evidence: Object.freeze([...row.evidence]) }); }
    evaluate({ taskClass, destination = null, dataScopes = [], effect = null, now = Date.now() } = {}) { const policy = this.policies.get(taskClass), confidence = this.score(taskClass); if (!policy)
        return Object.freeze({ authorized: false, reason: 'NO_STANDING_POLICY', confidence }); if (policy.expiresAt && now > policy.expiresAt)
        return Object.freeze({ authorized: false, reason: 'POLICY_EXPIRED', confidence, policy }); if (destination && policy.destinations.length && !policy.destinations.includes(destination))
        return Object.freeze({ authorized: false, reason: 'DESTINATION_OUT_OF_SCOPE', confidence, policy }); const outside = dataScopes.filter(scope => !policy.dataScopes.includes(scope)); if (outside.length)
        return Object.freeze({ authorized: false, reason: 'DATA_SCOPE_OUT_OF_SCOPE', outside: Object.freeze(outside), confidence, policy }); if (effect && policy.effects.length && !policy.effects.includes(effect))
        return Object.freeze({ authorized: false, reason: 'EFFECT_OUT_OF_SCOPE', confidence, policy }); if (effect === 'delete')
        return Object.freeze({ authorized: false, reason: 'DELETION_REQUIRES_SPECIFIC_REQUEST', confidence, policy }); return Object.freeze({ authorized: confidence.score >= policy.threshold, reason: confidence.score >= policy.threshold ? 'CONFIDENCE_THRESHOLD_MET' : 'CONFIDENCE_BELOW_THRESHOLD', confidence, policy }); }
    #event(type, payload) { this.events.push(Object.freeze({ sequence: ++this.sequence, type, payload })); }
    replay() { return Object.freeze([...this.events]); }
}
export class DeletionGuard {
    constructor() { this.requests = new Map(); this.sequence = 0; }
    request({ target, scope = 'local', reason = null } = {}) { if (!target)
        throw new AutonomyError('MISSING_DELETE_TARGET', 'Deletion request requires target'); const id = `delete-${++this.sequence}`; const request = Object.freeze({ id, target, scope, reason, status: 'requested' }); this.requests.set(id, request); return request; }
    confirm(id, { target } = {}) { const request = this.requests.get(id); if (!request)
        throw new AutonomyError('DELETE_REQUEST_NOT_FOUND', id); if (target !== request.target)
        throw new AutonomyError('DELETE_TARGET_MISMATCH', 'Confirmation must identify the same target'); const confirmed = Object.freeze({ ...request, status: 'confirmed' }); this.requests.set(id, confirmed); return confirmed; }
    authorize(id) { const request = this.requests.get(id); if (!request || request.status !== 'confirmed')
        return Object.freeze({ authorized: false, reason: 'SPECIFIC_CONFIRMED_REQUEST_REQUIRED' }); return Object.freeze({ authorized: true, request }); }
}
