import { normalizeArtifact } from './artifact-normalizer.mjs';
import { PrimitiveReducer } from './primitive-reducer.mjs';
import { FiveFieldCompiler } from './five-field-compiler.mjs';
import { OntologicalAddressSigner, digest } from './ontological-address.mjs';
import { BrowserAutomataStore } from './browser-automata-store.mjs';
import { BrowserExecutionSurface } from './browser-execution-surface.mjs';

function safeExecution(execution = {}) {
  return {
    ok: Boolean(execution.ok),
    kind: execution.kind ?? null,
    path: execution.path ?? null,
    backendUsed: execution.backendUsed ?? null,
    workerUsed: execution.workerUsed ?? null,
    error: execution.ok ? null : execution.error ?? null,
    unresolvedDependencies: [...(execution.unresolvedDependencies ?? [])],
    result: execution.result == null ? null : {
      engine: execution.result.engine ?? null,
      stdout: Array.isArray(execution.result.stdout) ? execution.result.stdout.slice(-128) : [],
      returnValue: execution.result.returnValue ?? null,
    },
  };
}

function buildPlan(decomposition, fieldMap) {
  const units = (decomposition.units ?? []).map((unit, index) => ({
    id: `unit:${index + 1}`,
    name: unit.name,
    kind: unit.kind,
    dependencies: [...(unit.structure?.dependencies ?? [])],
    moduleSyntax: Boolean(unit.structure?.moduleSyntax),
  }));
  return Object.freeze({
    schema: 'synthia.execution-plan.v1',
    workerRequired: false,
    backendRequired: false,
    units: Object.freeze(units.map(Object.freeze)),
    fieldOrder: fieldMap.order,
    fieldCounts: fieldMap.counts,
    unresolvedDependencies: Object.freeze(
      [...new Set(units.flatMap((unit) => unit.dependencies))],
    ),
  });
}

function makeNote({ fingerprint, reused, plan, execution, address }) {
  return Object.freeze({
    schema: 'synthia.execution-note.v1',
    fingerprint,
    reusedCompiledForm: reused,
    execution: safeExecution(execution),
    unresolvedDependencies: Object.freeze([...(execution.unresolvedDependencies ?? plan.unresolvedDependencies ?? [])]),
    addressId: address.id,
    nextRun: execution.ok
      ? 'reuse compiled decomposition and five-field plan; execute changed inputs only'
      : 'reuse compiled decomposition; resolve recorded dependencies before retry',
  });
}

export class AutomataIngestionExecutionKernel {
  constructor({
    reducer = new PrimitiveReducer(),
    compiler = new FiveFieldCompiler(),
    signer = new OntologicalAddressSigner(),
    store = new BrowserAutomataStore(),
    executionSurface = new BrowserExecutionSurface(),
  } = {}) {
    this.reducer = reducer;
    this.compiler = compiler;
    this.signer = signer;
    this.store = store;
    this.executionSurface = executionSurface;
    this.history = [];
  }

  async prepare(input, context = {}) {
    const normalized = await normalizeArtifact(input, context);
    const fingerprint = await digest(normalized.bytes);
    const cacheKey = `compiled:${fingerprint}`;
    const previous = await this.store.get(cacheKey);
    let decomposition;
    let fieldMap;
    let plan;
    let reused = false;

    if (previous?.decomposition && previous?.fieldMap && previous?.plan) {
      ({ decomposition, fieldMap, plan } = previous);
      reused = true;
    } else {
      decomposition = await this.reducer.reduce(normalized);
      fieldMap = this.compiler.compile(decomposition);
      plan = buildPlan(decomposition, fieldMap);
      await this.store.put(cacheKey, {
        schema: 'synthia.compiled-artifact.v1',
        fingerprint,
        artifact: decomposition.artifact,
        decomposition,
        fieldMap,
        plan,
      });
    }

    return Object.freeze({
      schema: 'synthia.automata-session.v1',
      fingerprint,
      normalized,
      decomposition,
      fieldMap,
      plan,
      reusedCompiledForm: reused,
    });
  }

  async finalize(session, execution = {}) {
    if (!session?.fingerprint) throw new TypeError('finalize requires a prepared automata session');
    const observedFields = this.compiler.observe(session.fieldMap, execution);
    const address = await this.signer.sign({
      fingerprint: session.fingerprint,
      fieldMap: observedFields,
      execution,
      artifact: session.decomposition.artifact,
    });
    const note = makeNote({
      fingerprint: session.fingerprint,
      reused: session.reusedCompiledForm,
      plan: session.plan,
      execution,
      address,
    });
    const record = Object.freeze({
      schema: 'synthia.ingestion-execution-record.v1',
      fingerprint: session.fingerprint,
      artifact: session.decomposition.artifact,
      reusedCompiledForm: session.reusedCompiledForm,
      decomposition: session.decomposition,
      fields: observedFields,
      plan: session.plan,
      execution: safeExecution(execution),
      address,
      note,
    });

    await this.store.put(`compiled:${session.fingerprint}`, {
      schema: 'synthia.compiled-artifact.v1',
      fingerprint: session.fingerprint,
      artifact: session.decomposition.artifact,
      decomposition: session.decomposition,
      fieldMap: session.fieldMap,
      plan: session.plan,
      lastAddress: address,
      lastNote: note,
    });
    await this.store.put(`address:${address.id}`, record);
    await this.store.put(`note:${session.fingerprint}`, note);
    this.history.push(record);
    return record;
  }

  async process(input, context = {}) {
    const session = await this.prepare(input, context);
    const execution = context.execute === false
      ? {
        ok: true,
        kind: session.normalized.kind,
        path: 'compiled-without-execution',
        backendUsed: false,
        workerUsed: false,
        result: { engine: 'synthia-automata-kernel', stdout: [], returnValue: null },
      }
      : await this.executionSurface.execute(session.normalized, context);
    return this.finalize(session, execution);
  }

  async recallByFingerprint(fingerprint) {
    return this.store.get(`compiled:${fingerprint}`);
  }

  async recallNote(fingerprint) {
    return this.store.get(`note:${fingerprint}`);
  }

  async recallAddress(addressId) {
    return this.store.get(`address:${addressId}`);
  }

  async audit() {
    return Object.freeze({
      kernel: 'synthia-automata-ingestion-execution',
      browserNative: true,
      workerRequired: false,
      backendRequiredForCore: false,
      fields: Object.freeze(['shared', 'knowledge', 'causal', 'dependency', 'stateSpace']),
      history: this.history.length,
      storage: await this.store.audit(),
    });
  }
}

export default AutomataIngestionExecutionKernel;
