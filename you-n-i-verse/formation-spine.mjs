// Integration slice: use the existing addressed state, ATO graph compiler,
// Anomata capability runtime and AutomataMesh without changing their contracts.
import { ATOEngine, FunctionRegistry } from './vendor/klein/index.mjs';
import { StateSpaceKernel } from './vendor/klein/state-space-kernel.mjs';
import { Anomata } from './vendor/klein/anomata.mjs';
import { Automaton, AutomataMesh } from './vendor/klein/automaton.mjs';

const copy = (value) => structuredClone(value);

/**
 * One logical formation, with versioned executable graphs.
 *
 * All revisions remain mounted and executable; transforming selects another
 * compiled graph, rather than relabeling or deleting the old implementation.
 * The journal is append-only within this runtime. Persistent external storage
 * is deliberately NOT claimed by this browser-local slice.
 */
export class ExecutableFormation {
  constructor({ id, address, graph, purpose = null,
    registry = new FunctionRegistry(), kernel = new StateSpaceKernel(),
    mesh = new AutomataMesh() } = {}) {
    if (typeof id !== 'string' || !id) throw new TypeError('Formation id required');
    this.id = id;
    this.kernel = kernel;
    this.address = kernel.get(address).address;
    this.purpose = purpose;
    this.engine = new ATOEngine({ registry });
    this.mesh = mesh;
    this.anomata = new Anomata({ kernel, maxEvents: Number.POSITIVE_INFINITY });
    this.revisions = [];
    this.activeRevision = null;
    this.#install(graph, 'formation-created', null);
  }

  #install(graph, eventType, reason) {
    // Compile first: invalid graph transformations never replace active code.
    const graphCopy = copy(graph);
    const number = this.revisions.length + 1;
    const revisionId = `${this.id}@${number}`;
    if (this.mesh.automatons.has(revisionId)) throw new Error(`Automaton already mounted: ${revisionId}`);
    const compiled = this.engine.compile({
      id: revisionId, address: this.address, purpose: this.purpose,
      graph: graphCopy,
    });
    const previousRevision = this.activeRevision;
    const automaton = new Automaton({
      id: revisionId, address: this.address,
      structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5],
      functionalLevel: 'being',
      ports: [
        { id: 'in', direction: 'input', type: 'json', schemaVersion: '1' },
        { id: 'out', direction: 'output', type: 'json', schemaVersion: '1' },
      ],
      metadata: {
        family: 'executable-formation', formationId: this.id,
        revision: number, previousRevision, graphArtifact: compiled.artifact.id,
      },
      implementation: (input) => this.anomata.run(revisionId, 'execute', input),
    });
    this.anomata.mount({
      id: revisionId, address: this.address,
      capabilities: [{
        id: 'execute', address: this.address, input: 'json', output: 'json',
        implementation: (input) => compiled.execute(input),
      }],
    });
    this.mesh.add(automaton);
    const revision = Object.freeze({
      id: revisionId, number, previousRevision, artifactId: compiled.artifact.id,
      graph: graphCopy, automaton,
    });
    this.revisions.push(revision);
    this.activeRevision = revisionId;
    this.kernel.annotate(this.address, 'code', {
      formationId: this.id, revisionId, artifactId: compiled.artifact.id,
      graph: copy(graphCopy),
    }, { source: 'executable-formation', reason });
    this.anomata.record(eventType, {
      formationId: this.id, revisionId, previousRevision, reason,
      artifactId: compiled.artifact.id, graph: copy(graphCopy),
    });
    return this.describe();
  }

  describe() {
    return Object.freeze({
      id: this.id,
      addressKey: this.kernel.get(this.address).addressKey,
      activeRevision: this.activeRevision,
      revisions: Object.freeze(this.revisions.map(({ id, number, previousRevision, artifactId, graph }) =>
        Object.freeze({ id, number, previousRevision, artifactId, graph: copy(graph) }))),
    });
  }

  transform(graph, { reason = 'structural-recomposition' } = {}) {
    try {
      return this.#install(graph, 'formation-transformed', reason);
    } catch (error) {
      this.anomata.record('formation-transform-rejected', {
        formationId: this.id, activeRevision: this.activeRevision,
        reason, error: String(error?.message ?? error),
      });
      throw error;
    }
  }

  async execute(input, { revisionId = this.activeRevision } = {}) {
    const revision = this.revisions.find((candidate) => candidate.id === revisionId);
    if (!revision) throw new Error(`Unknown formation revision: ${revisionId}`);
    try {
      const run = await this.mesh.run(revisionId, input);
      const result = run.outputs[revisionId].result;
      this.anomata.record('formation-executed', {
        formationId: this.id, revisionId, status: 'complete',
        input: copy(input), artifactId: result.artifactId,
        outputs: copy(result.outputs),
      });
      return result;
    } catch (error) {
      this.anomata.record('formation-executed', {
        formationId: this.id, revisionId, status: 'failed',
        input: copy(input), error: String(error?.message ?? error),
      });
      throw error;
    }
  }

  history() {
    return Object.freeze(this.anomata.replay()
      .filter(({ payload }) => payload?.formationId === this.id)
      .map((event) => Object.freeze(copy(event))));
  }
}
