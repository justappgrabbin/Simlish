import { DIMENSIONS } from './canonical-address.mjs';
import { executeDAG } from './dag-runtime.mjs';
import { stableStringify, fnv1a32 } from './pure-synthia/engine/derivation.js';

const ENCODER = new TextEncoder();

export const EXECUTION_PIPELINE_STAGES = Object.freeze([
  'ingest',
  'normalize',
  'primitive-operator-graph',
  'five-field-allocation',
  'dependency-schedule-forwarding',
  'artifact-execution',
  'fold-recombine',
  'state-commit',
  'observation',
  'ontological-address',
  'execution-cache',
]);

function safe(value) {
  try { return structuredClone(value); }
  catch {
    try { return JSON.parse(JSON.stringify(value)); }
    catch { return String(value); }
  }
}

function artifactBytes(artifact = {}) {
  if (artifact?.bytes instanceof Uint8Array) return artifact.bytes;
  if (artifact?.content instanceof Uint8Array) return artifact.content;
  if (typeof artifact?.content === 'string') return ENCODER.encode(artifact.content);
  if (typeof artifact?.originalContent === 'string') return ENCODER.encode(artifact.originalContent);
  return ENCODER.encode(stableStringify(artifact ?? null));
}

function normalizeArtifact(artifact = {}) {
  const bytes = artifactBytes(artifact);
  const name = artifact?.name ?? artifact?.originalName ?? artifact?.filename ?? 'artifact';
  const type = artifact?.type ?? null;
  return Object.freeze({
    name: String(name),
    type: type == null ? null : String(type),
    byteLength: bytes.length,
    fingerprint: fnv1a32(Array.from(bytes).join(',')),
  });
}

function emptyFieldSummary(field) {
  return {
    field,
    primitiveCount: 0,
    gateMin: null,
    gateMax: null,
    stateChecksum: 0,
    firstPrimitive: null,
    lastPrimitive: null,
  };
}

function fieldIndexForPrimitive(state, index) {
  // Runtime placement rule only. It is deliberately computational and makes no
  // scientific or metaphysical claim about the five dimensions.
  return (Number(state) + Number(index)) % DIMENSIONS.length;
}

function buildPrimitiveGraph(decomposition) {
  const states = [...(decomposition?.sixBitStates ?? [])];
  const nodes = [];
  let previousId = null;
  for (let index = 0; index < states.length; index += 1) {
    const state = states[index];
    const id = `primitive-${index}`;
    const field = DIMENSIONS[fieldIndexForPrimitive(state, index)];
    nodes.push(Object.freeze({
      id,
      index,
      sixBitState: state,
      bits: Number(state).toString(2).padStart(6, '0'),
      gate: Number(state) + 1,
      field,
      dependsOn: previousId ? Object.freeze([previousId]) : Object.freeze([]),
      dependencyKind: previousId ? 'source-order' : 'root',
    }));
    previousId = id;
  }
  return Object.freeze({
    nodeCount: nodes.length,
    dependencyCount: Math.max(0, nodes.length - 1),
    dependencySemantics: 'source-order-only',
    nodes: Object.freeze(nodes),
  });
}

function allocateFiveFields(graph) {
  const allocations = Object.fromEntries(DIMENSIONS.map((field) => [field, []]));
  for (const node of graph.nodes) allocations[node.field].push(node.id);
  return Object.freeze(Object.fromEntries(DIMENSIONS.map((field) => [
    field,
    Object.freeze({ field, primitiveIds: Object.freeze(allocations[field]) }),
  ])));
}

async function executePrimitiveSchedule(graph) {
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const accumulators = new Map(DIMENSIONS.map((field) => [field, emptyFieldSummary(field)]));
  const forwarding = [];

  const dag = graph.nodes.map((node) => ({ id: node.id, dependsOn: [...node.dependsOn] }));
  const result = await executeDAG(dag, async (dagNode, dagContext) => {
    const node = byId.get(dagNode.id);
    const field = accumulators.get(node.field);
    field.primitiveCount += 1;
    field.gateMin = field.gateMin == null ? node.gate : Math.min(field.gateMin, node.gate);
    field.gateMax = field.gateMax == null ? node.gate : Math.max(field.gateMax, node.gate);
    field.stateChecksum = (((field.stateChecksum * 33) ^ node.sixBitState ^ node.index) >>> 0);
    field.firstPrimitive ??= node.id;
    field.lastPrimitive = node.id;

    let forwarded = null;
    const predecessorId = node.dependsOn[0] ?? null;
    if (predecessorId) {
      const predecessor = byId.get(predecessorId);
      if (predecessor && predecessor.field !== node.field) {
        const priorResult = dagContext.dependencies[predecessorId] ?? null;
        forwarded = Object.freeze({
          fromPrimitive: predecessorId,
          toPrimitive: node.id,
          fromField: predecessor.field,
          toField: node.field,
          value: priorResult ? safe(priorResult.outputState) : null,
          reason: 'source-order-dependency',
        });
        forwarding.push(forwarded);
      }
    }

    return Object.freeze({
      primitiveId: node.id,
      field: node.field,
      outputState: Object.freeze({
        gate: node.gate,
        sixBitState: node.sixBitState,
        bits: node.bits,
        field: node.field,
      }),
      forwarded,
    });
  }, { concurrency: DIMENSIONS.length, continueOnError: true });

  const fieldSummaries = Object.freeze(Object.fromEntries(DIMENSIONS.map((field) => [
    field,
    Object.freeze({ ...accumulators.get(field) }),
  ])));

  return Object.freeze({
    ok: result.ok,
    status: Object.freeze({ ...result.status }),
    errors: Object.freeze({ ...result.errors }),
    fieldSummaries,
    forwarding: Object.freeze(forwarding),
    forwardingCount: forwarding.length,
    scheduledPrimitives: graph.nodeCount,
  });
}

export class MemoryExecutionPipelineStore {
  constructor(initial = null) {
    this.value = initial == null ? null : safe(initial);
  }
  async load() { return this.value == null ? null : safe(this.value); }
  async save(value) { this.value = safe(value); return true; }
}

export class LocalStorageExecutionPipelineStore {
  constructor({ key = 'synthia.execution.pipeline.v1', storage = globalThis?.localStorage } = {}) {
    if (!storage || typeof storage.getItem !== 'function' || typeof storage.setItem !== 'function') {
      throw new TypeError('LocalStorageExecutionPipelineStore requires a Web Storage compatible object');
    }
    this.key = key;
    this.storage = storage;
  }
  async load() {
    const raw = this.storage.getItem(this.key);
    return raw ? JSON.parse(raw) : null;
  }
  async save(value) {
    this.storage.setItem(this.key, JSON.stringify(value));
    return true;
  }
}

export function createDefaultExecutionPipelineStore() {
  try {
    if (typeof globalThis !== 'undefined' && globalThis.localStorage) {
      return new LocalStorageExecutionPipelineStore({ storage: globalThis.localStorage });
    }
  } catch {
    // Access to browser storage can itself throw in hardened/sandboxed contexts.
    // Falling back to memory is explicit in the returned store type and audit.
  }
  return new MemoryExecutionPipelineStore();
}

export class ExecutionPipeline {
  constructor({
    executeArtifactCore,
    addressResolver,
    mesh = null,
    store = null,
    cacheLimit = 16,
  } = {}) {
    if (typeof executeArtifactCore !== 'function') {
      throw new TypeError('ExecutionPipeline requires executeArtifactCore(artifact, context)');
    }
    if (!addressResolver || typeof addressResolver.descend !== 'function') {
      throw new TypeError('ExecutionPipeline requires an address resolver with descend()');
    }
    this.executeArtifactCore = executeArtifactCore;
    this.addressResolver = addressResolver;
    this.mesh = mesh;
    this.store = store ?? createDefaultExecutionPipelineStore();
    this.cacheLimit = Math.max(1, Number(cacheLimit) || 16);
    this.analysisCache = new Map();
    this.fieldState = new Map(DIMENSIONS.map((field) => [field, {
      field,
      runs: 0,
      totalPrimitives: 0,
      lastFingerprint: null,
      lastChecksum: 0,
      lastCanonicalAddressKey: null,
    }]));
    this.commits = [];
    this.runSequence = 0;
    this.loaded = false;
  }

  async #ensureLoaded() {
    if (this.loaded) return;
    const snapshot = await this.store.load();
    if (snapshot && snapshot.version === 1) {
      this.runSequence = Number(snapshot.runSequence) || 0;
      for (const entry of snapshot.fields ?? []) {
        if (entry && DIMENSIONS.includes(entry.field)) this.fieldState.set(entry.field, { ...entry });
      }
      for (const entry of snapshot.cache ?? []) {
        if (entry?.fingerprint && entry?.analysis) this.analysisCache.set(entry.fingerprint, entry.analysis);
      }
      this.commits = Array.isArray(snapshot.commits) ? snapshot.commits.slice(-64).map(safe) : [];
    }
    this.loaded = true;
  }

  #touchCache(fingerprint, analysis) {
    if (this.analysisCache.has(fingerprint)) this.analysisCache.delete(fingerprint);
    this.analysisCache.set(fingerprint, safe(analysis));
    while (this.analysisCache.size > this.cacheLimit) {
      const oldest = this.analysisCache.keys().next().value;
      this.analysisCache.delete(oldest);
    }
  }

  async #persist() {
    const snapshot = {
      version: 1,
      runSequence: this.runSequence,
      fields: DIMENSIONS.map((field) => safe(this.fieldState.get(field))),
      cache: [...this.analysisCache.entries()].map(([fingerprint, analysis]) => ({ fingerprint, analysis: safe(analysis) })),
      commits: this.commits.slice(-64).map(safe),
    };
    await this.store.save(snapshot);
    return snapshot;
  }

  #commitToMesh(commit, analysis) {
    if (!this.mesh || typeof this.mesh.addEdge !== 'function') return { written: 0 };
    let written = 0;
    const executionNode = `execution:${commit.sequence}:${commit.fingerprint}`;
    for (const field of DIMENSIONS) {
      const summary = analysis.schedule.fieldSummaries[field];
      if (!summary.primitiveCount) continue;
      this.mesh.addEdge('knowledge', executionNode, `field:${field}`, 'allocated-to', {
        primitiveCount: summary.primitiveCount,
        checksum: summary.stateChecksum,
      });
      written += 1;
    }
    for (const edge of analysis.schedule.forwarding) {
      this.mesh.addEdge('dependency', `field:${edge.fromField}`, `field:${edge.toField}`, 'forwards', {
        fromPrimitive: edge.fromPrimitive,
        toPrimitive: edge.toPrimitive,
        reason: edge.reason,
      });
      written += 1;
    }
    const prior = this.commits.length > 1 ? this.commits[this.commits.length - 2] : null;
    if (prior) {
      this.mesh.addEdge('temporal', `execution:${prior.sequence}:${prior.fingerprint}`, executionNode, 'next-execution', {});
      written += 1;
    }
    return { written };
  }

  #fold(analysis, coreResult) {
    const fields = DIMENSIONS.map((field) => analysis.schedule.fieldSummaries[field]);
    return Object.freeze({
      fieldOrder: Object.freeze([...DIMENSIONS]),
      fields: Object.freeze(fields.map((entry) => Object.freeze({ ...entry }))),
      forwardingCount: analysis.schedule.forwardingCount,
      runtime: Object.freeze({
        ok: Boolean(coreResult.ok),
        path: coreResult.path ?? null,
        kind: coreResult.kind ?? null,
        runtimeAdapter: coreResult.runtimeAdapter ?? null,
      }),
      canonicalAddressKey: coreResult.canonicalAddressKey ?? null,
      foldHash: fnv1a32(stableStringify({
        fields,
        runtime: {
          ok: Boolean(coreResult.ok),
          path: coreResult.path ?? null,
          kind: coreResult.kind ?? null,
          runtimeAdapter: coreResult.runtimeAdapter ?? null,
        },
        canonicalAddressKey: coreResult.canonicalAddressKey ?? null,
      })),
    });
  }

  async execute(artifact, context = {}) {
    await this.#ensureLoaded();
    const normalized = normalizeArtifact(artifact);
    const fingerprint = normalized.fingerprint;
    let cacheHit = false;
    let analysis = null;

    if (context.forceReanalyze !== true) {
      const cached = this.analysisCache.get(fingerprint);
      if (cached) {
        cacheHit = true;
        analysis = safe(cached);
        this.#touchCache(fingerprint, analysis);
      }
    }

    if (!analysis) {
      const decomposition = this.addressResolver.descend(artifact);
      const primitiveGraph = buildPrimitiveGraph(decomposition);
      const allocation = allocateFiveFields(primitiveGraph);
      const schedule = await executePrimitiveSchedule(primitiveGraph);
      if (!schedule.ok) {
        const error = new Error('primitive dependency schedule failed');
        error.pipelineSchedule = safe(schedule);
        throw error;
      }
      analysis = {
        normalized: safe(normalized),
        decomposition: safe(decomposition),
        primitiveGraph: safe(primitiveGraph),
        allocation: safe(allocation),
        schedule: safe(schedule),
      };
      this.#touchCache(fingerprint, analysis);
    }

    const coreResult = await this.executeArtifactCore(artifact, {
      ...context,
      executionDecomposition: safe(analysis.decomposition),
      executionPipeline: {
        fingerprint,
        primitiveGraph: safe(analysis.primitiveGraph),
        allocation: safe(analysis.allocation),
        schedule: safe(analysis.schedule),
        analysisCacheHit: cacheHit,
      },
    });

    const fold = this.#fold(analysis, coreResult);
    const sequence = ++this.runSequence;
    for (const field of DIMENSIONS) {
      const prior = this.fieldState.get(field) ?? { field, runs: 0, totalPrimitives: 0 };
      const current = analysis.schedule.fieldSummaries[field];
      this.fieldState.set(field, {
        ...prior,
        field,
        runs: (prior.runs ?? 0) + 1,
        totalPrimitives: (prior.totalPrimitives ?? 0) + current.primitiveCount,
        lastFingerprint: fingerprint,
        lastChecksum: current.stateChecksum,
        lastCanonicalAddressKey: coreResult.canonicalAddressKey ?? null,
      });
    }

    const commit = Object.freeze({
      sequence,
      fingerprint,
      ok: Boolean(coreResult.ok),
      canonicalAddressKey: coreResult.canonicalAddressKey ?? null,
      foldHash: fold.foldHash,
      forwardingCount: analysis.schedule.forwardingCount,
      primitiveCount: analysis.primitiveGraph.nodeCount,
      cacheHit,
    });
    this.commits.push(commit);
    if (this.commits.length > 64) this.commits.splice(0, this.commits.length - 64);

    const meshCommit = this.#commitToMesh(commit, analysis);
    const persisted = await this.#persist();

    const stageRecords = Object.freeze([
      { stage: 'ingest', ok: true, evidence: { name: normalized.name, byteLength: normalized.byteLength } },
      { stage: 'normalize', ok: true, evidence: { fingerprint } },
      { stage: 'primitive-operator-graph', ok: true, evidence: { nodes: analysis.primitiveGraph.nodeCount, dependencies: analysis.primitiveGraph.dependencyCount, semantics: analysis.primitiveGraph.dependencySemantics } },
      { stage: 'five-field-allocation', ok: true, evidence: Object.fromEntries(DIMENSIONS.map((field) => [field, analysis.schedule.fieldSummaries[field].primitiveCount])) },
      { stage: 'dependency-schedule-forwarding', ok: analysis.schedule.ok, evidence: { scheduled: analysis.schedule.scheduledPrimitives, forwarding: analysis.schedule.forwardingCount } },
      { stage: 'artifact-execution', ok: Boolean(coreResult.ok), evidence: { path: coreResult.path ?? null, kind: coreResult.kind ?? null, runtimeAdapter: coreResult.runtimeAdapter ?? null } },
      { stage: 'fold-recombine', ok: true, evidence: { foldHash: fold.foldHash } },
      { stage: 'state-commit', ok: true, evidence: { sequence, meshEdgesWritten: meshCommit.written } },
      { stage: 'observation', ok: true, evidence: { recordId: coreResult.recordId ?? null, placementRecordId: coreResult.placementRecordId ?? null } },
      { stage: 'ontological-address', ok: Boolean(coreResult.canonicalAddressKey), evidence: { canonicalAddressKey: coreResult.canonicalAddressKey ?? null, basis: coreResult.addressBasis ?? null } },
      { stage: 'execution-cache', ok: true, evidence: { analysisCacheHit: cacheHit, entries: this.analysisCache.size, persisted: Boolean(persisted) } },
    ]);

    return {
      ...coreResult,
      pipeline: Object.freeze({
        version: 'execution-pipeline.v1',
        fingerprint,
        cache: Object.freeze({ analysisHit: cacheHit, entries: this.analysisCache.size }),
        primitiveGraph: safe(analysis.primitiveGraph),
        allocation: safe(analysis.allocation),
        schedule: safe(analysis.schedule),
        fold,
        commit,
        fields: Object.freeze(Object.fromEntries(DIMENSIONS.map((field) => [field, Object.freeze({ ...this.fieldState.get(field) })]))),
        stages: stageRecords,
      }),
    };
  }

  snapshot() {
    return Object.freeze({
      version: 'execution-pipeline.v1',
      runSequence: this.runSequence,
      cacheEntries: this.analysisCache.size,
      fields: Object.freeze(Object.fromEntries(DIMENSIONS.map((field) => [field, Object.freeze({ ...this.fieldState.get(field) })]))),
      commits: Object.freeze(this.commits.map((entry) => Object.freeze({ ...entry }))),
      store: this.store?.constructor?.name ?? 'unknown',
    });
  }

  wiringAudit() {
    const snapshot = this.snapshot();
    return Object.freeze({
      ok: Boolean(
        typeof this.executeArtifactCore === 'function' &&
        this.addressResolver &&
        this.store &&
        DIMENSIONS.every((field) => snapshot.fields[field])
      ),
      stages: EXECUTION_PIPELINE_STAGES,
      fiveFields: Object.freeze([...DIMENSIONS]),
      store: snapshot.store,
      cacheEntries: snapshot.cacheEntries,
      runSequence: snapshot.runSequence,
      committedRuns: snapshot.commits.length,
    });
  }
}

export default ExecutionPipeline;
