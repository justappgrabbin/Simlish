import { Automaton } from './automaton.mjs';
export class BrowserFormError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'BrowserFormError'; this.code = code; this.details = details; }
}
const SENSITIVE = new Set(['ssn', 'social-security-number', 'password', 'pin', 'bank-account', 'routing-number', 'medical', 'immigration-status']);
export class BrowserFormState {
    constructor() { this.page = null; this.forms = new Map(); this.drafts = new Map(); this.approvals = new Map(); this.events = []; this.sequence = 0; }
    inspectPage({ url, title = null, forms = [] } = {}) { if (!url)
        throw new BrowserFormError('URL_REQUIRED', 'Page inspection requires URL'); this.page = Object.freeze({ url, title, forms: Object.freeze(forms.map(f => f.id)) }); for (const form of forms)
        this.inspectForm(form); this.#event('page-inspected', this.page); return this.snapshot(); }
    inspectForm({ id, action, method = 'POST', fields = [], official = false } = {}) { if (!id)
        throw new BrowserFormError('FORM_ID_REQUIRED', 'Form requires id'); const form = Object.freeze({ id, action, method, official, fields: Object.freeze(fields.map((field, index) => Object.freeze({ id: field.id || field.name || `field-${index + 1}`, name: field.name || field.id, label: field.label || field.name, type: field.type || 'text', required: Boolean(field.required), sensitivity: field.sensitivity || (SENSITIVE.has(field.name) ? 'high' : 'normal'), options: Object.freeze([...(field.options || [])]), value: field.value ?? null }))) }); this.forms.set(id, form); this.#event('form-inspected', form); return form; }
    createDraft(formId, values, { sources = {} } = {}) { const form = this.forms.get(formId); if (!form)
        throw new BrowserFormError('FORM_NOT_FOUND', formId); const entries = form.fields.map(field => { const provided = Object.prototype.hasOwnProperty.call(values, field.name), source = sources[field.name] || null; return Object.freeze({ fieldId: field.id, name: field.name, label: field.label, value: provided ? values[field.name] : null, status: provided ? (source?.kind === 'inferred' ? 'inferred' : 'resolved') : 'unresolved', sensitivity: field.sensitivity, provenance: source ? Object.freeze({ ...source }) : null, approved: false }); }); const id = `draft-${++this.sequence}-${formId}`; const draft = Object.freeze({ id, formId, destination: form.action, entries: Object.freeze(entries), status: 'draft' }); this.drafts.set(id, draft); this.events.push(Object.freeze({ sequence: this.sequence, type: 'draft-created', payload: draft })); return draft; }
    approveFields(draftId, names, { scope = 'once' } = {}) { const draft = this.#draft(draftId), set = new Set(names); const missing = names.filter(name => !draft.entries.some(entry => entry.name === name)); if (missing.length)
        throw new BrowserFormError('FIELD_NOT_FOUND', 'Unknown draft fields', { missing }); const updated = Object.freeze({ ...draft, entries: Object.freeze(draft.entries.map(entry => set.has(entry.name) ? Object.freeze({ ...entry, approved: true }) : entry)), status: 'fields-approved' }); this.drafts.set(draftId, updated); this.approvals.set(`fields:${draftId}`, Object.freeze({ draftId, names: Object.freeze([...set]), scope })); this.#event('fields-approved', { draftId, names: [...set] }); return updated; }
    fill(draftId) { const draft = this.#draft(draftId), unapproved = draft.entries.filter(entry => entry.value !== null && !entry.approved); if (unapproved.length)
        throw new BrowserFormError('FIELD_APPROVAL_REQUIRED', 'Every populated value requires approval', { fields: unapproved.map(x => x.name) }); const inferred = draft.entries.filter(entry => entry.status === 'inferred'); if (inferred.length)
        throw new BrowserFormError('INFERRED_VALUE_CONFIRMATION_REQUIRED', 'Inferred values must be confirmed', { fields: inferred.map(x => x.name) }); const updated = Object.freeze({ ...draft, status: 'filled' }); this.drafts.set(draftId, updated); this.#event('form-filled', { draftId }); return updated; }
    confirmInferred(draftId, names) { const draft = this.#draft(draftId), set = new Set(names), updated = Object.freeze({ ...draft, entries: Object.freeze(draft.entries.map(entry => set.has(entry.name) && entry.status === 'inferred' ? Object.freeze({ ...entry, status: 'resolved', approved: true, provenance: Object.freeze({ ...entry.provenance, confirmed: true }) }) : entry)) }); this.drafts.set(draftId, updated); this.#event('inferred-confirmed', { draftId, names: [...set] }); return updated; }
    validate(draftId) { const draft = this.#draft(draftId), form = this.forms.get(draft.formId), missing = form.fields.filter(field => field.required && !draft.entries.some(entry => entry.name === field.name && entry.value !== null && entry.value !== '')).map(field => field.name); const result = Object.freeze({ valid: missing.length === 0, missing: Object.freeze(missing), draftId }); this.#event('form-validated', result); return result; }
    requestSubmission(draftId) { const draft = this.#draft(draftId); if (draft.status !== 'filled')
        throw new BrowserFormError('FORM_NOT_FILLED', 'Submission requires filled draft'); const validation = this.validate(draftId); if (!validation.valid)
        throw new BrowserFormError('VALIDATION_FAILED', 'Required fields unresolved', { missing: validation.missing }); const request = Object.freeze({ id: `submit-${++this.sequence}-${draftId}`, draftId, destination: draft.destination, fields: Object.freeze(draft.entries.filter(e => e.value !== null).map(e => Object.freeze({ name: e.name, value: e.value, sensitivity: e.sensitivity, provenance: e.provenance }))), status: 'awaiting-confirmation' }); this.approvals.set(request.id, request); this.events.push(Object.freeze({ sequence: this.sequence, type: 'submission-requested', payload: request })); return request; }
    confirmSubmission(requestId) { const request = this.approvals.get(requestId); if (!request || request.status !== 'awaiting-confirmation')
        throw new BrowserFormError('SUBMISSION_REQUEST_NOT_FOUND', requestId); const confirmed = Object.freeze({ ...request, status: 'confirmed' }); this.approvals.set(requestId, confirmed); this.#event('submission-confirmed', { requestId }); return confirmed; }
    submit(requestId, executor) { const request = this.approvals.get(requestId); if (!request || request.status !== 'confirmed')
        throw new BrowserFormError('SUBMISSION_CONFIRMATION_REQUIRED', 'External submission needs final confirmation'); if (typeof executor !== 'function')
        throw new BrowserFormError('BROWSER_EXECUTOR_REQUIRED', 'Submission requires mounted browser executor'); this.#event('submission-dispatched', { requestId, destination: request.destination }); return executor(request); }
    #draft(id) { const draft = this.drafts.get(id); if (!draft)
        throw new BrowserFormError('DRAFT_NOT_FOUND', id); return draft; }
    #event(type, payload) { const event = Object.freeze({ sequence: ++this.sequence, type, payload }); this.events.push(event); return event; }
    snapshot() { return Object.freeze({ page: this.page, forms: Object.freeze([...this.forms.values()]), drafts: Object.freeze([...this.drafts.values()]), events: Object.freeze([...this.events]) }); }
}
export function browserFormAutomaton({ state = new BrowserFormState() } = {}) { return new Automaton({ id: 'browser-form', address: { mode: 'macro', gate: 20, line: 1, color: 1, tone: 1, base: 1 }, structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5], functionalLevel: 'being', ports: [{ id: 'request', direction: 'input', type: 'browser-form', schemaVersion: '1' }, { id: 'state', direction: 'output', type: 'browser-form', schemaVersion: '1', guarantees: ['addressed'] }], state, metadata: { family: 'browser-form', selfReferential: true, consentGated: true }, implementation: (input, { state }) => { const op = input.operation; if (op === 'inspect-page')
        return state.inspectPage(input.page); if (op === 'draft')
        return state.createDraft(input.formId, input.values, input.context); if (op === 'approve-fields')
        return state.approveFields(input.draftId, input.names, input.context); if (op === 'confirm-inferred')
        return state.confirmInferred(input.draftId, input.names); if (op === 'fill')
        return state.fill(input.draftId); if (op === 'validate')
        return state.validate(input.draftId); if (op === 'request-submission')
        return state.requestSubmission(input.draftId); if (op === 'confirm-submission')
        return state.confirmSubmission(input.requestId); if (op === 'snapshot')
        return state.snapshot(); throw new BrowserFormError('UNKNOWN_OPERATION', op); } }); }
