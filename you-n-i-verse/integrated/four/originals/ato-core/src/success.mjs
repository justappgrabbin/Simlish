import { Automaton } from './automaton.mjs';

export class SuccessError extends Error {
  constructor(code, message, details = {}) { super(message); this.name = 'SuccessError'; this.code = code; this.details = details; }
}

const trend = (values) => {
  if (values.length < 2) return 0;
  const n = values.length;
  const sumX = values.reduce((sum, _, index) => sum + index, 0);
  const sumY = values.reduce((sum, value) => sum + value, 0);
  const sumXY = values.reduce((sum, value, index) => sum + index * value, 0);
  const sumX2 = values.reduce((sum, _, index) => sum + index * index, 0);
  const denominator = n * sumX2 - sumX * sumX;
  return denominator === 0 ? 0 : (n * sumXY - sumX * sumY) / denominator;
};

export class SuccessLedger {
  constructor({ sequence = 0 } = {}) {
    this.sequence = sequence;
    this.people = new Map();
    this.events = [];
  }

  define(personId, { statement, indicators = [], address = null } = {}) {
    if (!personId || !statement?.trim()) throw new SuccessError('INVALID_PURPOSE', 'Person and purpose statement are required');
    const purpose = Object.freeze({
      personId,
      statement: statement.trim(),
      address,
      revision: 1,
      indicators: Object.freeze(indicators.map((indicator, index) => Object.freeze({
        id: indicator.id || `indicator-${index + 1}`,
        name: indicator.name,
        measure: indicator.measure || null,
        direction: indicator.direction || 'increase',
        weight: Number.isFinite(indicator.weight) ? indicator.weight : 1,
      }))),
    });
    this.people.set(personId, { purpose, observations: new Map(), proposals: [] });
    this.#record('purpose-defined', { personId, purpose });
    return purpose;
  }

  observe(personId, indicatorId, value, { context = null, address = null, source = 'person' } = {}) {
    const person = this.people.get(personId);
    if (!person) throw new SuccessError('PURPOSE_NOT_FOUND', personId);
    const indicator = person.purpose.indicators.find((item) => item.id === indicatorId);
    if (!indicator) throw new SuccessError('INDICATOR_NOT_FOUND', indicatorId);
    if (!Number.isFinite(value)) throw new SuccessError('INVALID_OBSERVATION', 'Progress observation must be numeric');
    const history = person.observations.get(indicatorId) || [];
    const observation = Object.freeze({ sequence: ++this.sequence, value, context, address, source });
    history.push(observation);
    person.observations.set(indicatorId, history);
    const progress = this.progress(personId, indicatorId);
    this.events.push(Object.freeze({ sequence: this.sequence, type: 'progress-observed', payload: { personId, indicatorId, observation, progress } }));
    return progress;
  }

  progress(personId, indicatorId) {
    const person = this.people.get(personId);
    if (!person) throw new SuccessError('PURPOSE_NOT_FOUND', personId);
    const indicator = person.purpose.indicators.find((item) => item.id === indicatorId);
    if (!indicator) throw new SuccessError('INDICATOR_NOT_FOUND', indicatorId);
    const history = person.observations.get(indicatorId) || [];
    const values = history.slice(-5).map((item) => item.value);
    const rate = trend(values);
    let direction = 'unknown';
    if (values.length >= 2) {
      if (indicator.direction === 'increase') direction = rate > 0 ? 'toward' : rate < 0 ? 'away' : 'neutral';
      if (indicator.direction === 'decrease') direction = rate < 0 ? 'toward' : rate > 0 ? 'away' : 'neutral';
      if (indicator.direction === 'maintain') direction = Math.abs(rate) <= 0.1 ? 'toward' : 'away';
    }
    return Object.freeze({ personId, indicatorId, direction, rate: Math.abs(rate), signedRate: rate, values: Object.freeze(values), evidenceCount: history.length });
  }

  propose(personId, indicatorId, { message, behaviors = [], role = 'companion', address = null } = {}) {
    const person = this.people.get(personId);
    if (!person) throw new SuccessError('PURPOSE_NOT_FOUND', personId);
    const progress = this.progress(personId, indicatorId);
    const proposal = Object.freeze({ id: `success-${personId}-${++this.sequence}`, personId, indicatorId, progress, role, message, behaviors: Object.freeze([...behaviors]), address, status: 'proposed' });
    person.proposals.push(proposal);
    this.events.push(Object.freeze({ sequence: this.sequence, type: 'success-proposal', payload: proposal }));
    return proposal;
  }

  summary(personId) {
    const person = this.people.get(personId);
    if (!person) return null;
    return Object.freeze({
      purpose: person.purpose,
      progress: Object.freeze(person.purpose.indicators.map((item) => this.progress(personId, item.id))),
      proposals: Object.freeze([...person.proposals]),
    });
  }

  #record(type, payload) { this.events.push(Object.freeze({ sequence: ++this.sequence, type, payload })); }
  replay() { return Object.freeze([...this.events]); }
}

export function successAutomaton({ id = 'success', address = { mode: 'macro', gate: 14, line: 1, color: 1, tone: 1, base: 1 }, ledger = new SuccessLedger() } = {}) {
  return new Automaton({
    id,
    address,
    structure: 'hexagram',
    activeLevels: [1, 2, 3, 4, 5],
    functionalLevel: 'being',
    ports: [
      { id: 'purpose', direction: 'input', type: 'success', schemaVersion: '1', requires: ['addressed'], guarantees: [] },
      { id: 'expression', direction: 'output', type: 'success', schemaVersion: '1', requires: [], guarantees: ['addressed'] },
    ],
    state: ledger,
    metadata: { family: 'success', donorLineage: ['SuccessDrivenPurposeCore'], sourceMode: 'direction-not-destination' },
    implementation: (input, { state }) => {
      if (input.type === 'define') return state.define(input.personId, input.purpose);
      if (input.type === 'observe') return state.observe(input.personId, input.indicatorId, input.value, input.context);
      if (input.type === 'propose') return state.propose(input.personId, input.indicatorId, input.proposal);
      if (input.type === 'summary') return state.summary(input.personId);
      throw new SuccessError('UNKNOWN_SUCCESS_OPERATION', input.type);
    },
  });
}
