/**
 * Mounts the REAL recovered AUTOLING and Diseminer automatons as operations
 * in the Computer's existing ATO FunctionRegistry.
 *
 * The append-only Computer event emitter is the durable source of donor memory
 * changes. Replaying native Diseminer.ingest() reconstructs actual co-occurrence
 * vectors and documents; no substitute language parser or fake vectors.
 */
import { AutoLingMemory, autoLingAutomaton, diseminerAutomaton } from
  '../donors/recovered/synthai-r21.22-self-cultivation-graph-integrated/vendor/ato-core/src/klein-tools.mjs';

export const KLEIN_MEMORY_PROVIDER = 'synthai:recovered-klein-memory';
export class NativeKleinMemory {
  constructor({ computer } = {}) {
    if (!computer?.formations || !computer?.eventEmitter)
      throw new TypeError('NativeKleinMemory requires the Computer formation service and event ledger');
    this.computer = computer;
    this.autoLing = autoLingAutomaton();
    this.diseminer = diseminerAutomaton();
    this.seen = new Set();
    this.pending = Promise.resolve();
  }

  async #apply(event) {
    if (event.provider !== KLEIN_MEMORY_PROVIDER) return;
    if (this.seen.has(event.event_id)) return;
    if (event.result_status !== 'success')
      throw new Error('Invalid native Klein memory history: ' + event.event_id);
    if (event.input?.operation === 'ingest') {
      await this.diseminer.call({
        operation: 'ingest', text: event.input.text,
        context: { source: event.input.source, address: event.actor_address },
      });
    } else if (event.input?.operation === 'learn') {
      await this.autoLing.call({ operation: 'learn', rule: event.input.rule });
    } else {
      throw new Error('Unknown native Klein memory event: ' + event.event_id);
    }
    this.seen.add(event.event_id);
  }

  async #refresh() {
    for (const event of await this.computer.eventEmitter.readAll()) await this.#apply(event);
  }

  async boot() {
    await this.#refresh();
    this.computer.registerFormationOperation(
      'klein.autoling.recognize',
      async input => {
        await this.#refresh();
        return this.autoLing.call({ operation: 'recognize', text: input?.text ?? String(input ?? '') });
      },
    );
    this.computer.registerFormationOperation(
      'klein.diseminer.ingest-recognized',
      recognized => this.ingestRecognized(recognized),
    );
    return this;
  }

  async learnRule(rule) {
    if (!rule || typeof rule.id !== 'string' || !rule.id ||
      typeof rule.relation !== 'string' || !rule.relation ||
      !Array.isArray(rule.pattern) || !rule.pattern.length ||
      !rule.pattern.every(token => typeof token === 'string' && token.length > 0) ||
      typeof rule.template !== 'string' || !rule.template.length)
      throw new TypeError('AUTOLING learning requires an id, relation, token pattern, and template');
    // The real donor validates semantics before any committed event is written.
    new AutoLingMemory().learn(rule);
    const work = async () => {
      await this.#refresh();
      if (this.autoLing.ownedState.rules.has(rule.id))
        throw new Error('AUTOLING rule ID already exists: ' + rule.id);
      const event = await this.computer.emitEvent({
        actor_id: 'automaton:autoling', actor_type: 'automaton',
        target_id: 'klein:autoling', event_type: 'tool_execution',
        provider: KLEIN_MEMORY_PROVIDER,
        trajectory_id: 'klein:autoling-memory',
        input: { operation: 'learn', rule: structuredClone(rule) },
        result_status: 'success',
        observable_effect: 'AUTOLING native grammar rule committed',
      });
      await this.#apply(event);
      return this.autoLing.ownedState.rules.get(rule.id);
    };
    const next = this.pending.then(work);
    this.pending = next.then(() => undefined, () => undefined);
    return next;
  }

  async ingestRecognized(recognized) {
    if (!recognized?.ok || !recognized?.matches?.length)
      return Object.freeze({ ok: false, reason: 'NO_AUTOLING_MATCH' });
    const text = recognized.text;
    // Serialize native memory operations within this Computer instance.
    const work = async () => {
      await this.#refresh();
      const event = await this.computer.emitEvent({
        actor_id: 'automaton:diseminer', actor_type: 'automaton',
        target_id: 'klein:diseminer', event_type: 'tool_execution',
        provider: KLEIN_MEMORY_PROVIDER,
        trajectory_id: 'klein:diseminer-memory',
        input: { operation: 'ingest', text, source: 'autoling-formation' },
        result_status: 'success',
        observable_effect: 'Diseminer native lexical memory ingest committed',
      });
      await this.#apply(event);
      const doc = this.diseminer.ownedState.documents.at(-1);
      return Object.freeze({
        ok: true,
        relation: recognized.matches[0].relation,
        bindings: recognized.matches[0].bindings,
        document: doc,
      });
    };
    const next = this.pending.then(work);
    this.pending = next.then(() => undefined, () => undefined);
    return next;
  }

  async snapshot() {
    await this.pending;
    await this.#refresh();
    const exported = await this.diseminer.call({ operation: 'export' });
    const rules = await this.autoLing.call({ operation: 'export' });
    return Object.freeze({
      provider: KLEIN_MEMORY_PROVIDER,
      documentCount: exported.documents.length,
      terms: Object.freeze(Object.keys(exported.vectors).sort()),
      documents: exported.documents,
      rules: rules,
    });
  }
}
