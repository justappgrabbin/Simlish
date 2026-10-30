export const EXECUTION_FIELDS = Object.freeze(['shared', 'knowledge', 'causal', 'dependency', 'stateSpace']);

function dedupeRecords(records) {
  const seen = new Set();
  const out = [];
  for (const record of records) {
    const key = JSON.stringify(record);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(Object.freeze(record));
  }
  return Object.freeze(out);
}

function fieldRecord(kind, value, source) {
  return { kind, value, source };
}

export class FiveFieldCompiler {
  compile(decomposition) {
    const shared = [];
    const knowledge = [];
    const causal = [];
    const dependency = [];
    const stateSpace = [];

    const allUnits = decomposition.root?.kind === 'zip'
      ? [decomposition.root, ...(decomposition.units ?? [])]
      : (decomposition.units ?? [decomposition.root]).filter(Boolean);

    for (const unit of allUnits) {
      const source = unit.name ?? decomposition.artifact?.name ?? 'artifact';
      shared.push(fieldRecord('artifact-kind', unit.kind, source));
      for (const primitive of unit.primitives ?? []) {
        if (['js-operator', 'artifact-kind', 'text-token'].includes(primitive.type)) {
          shared.push(fieldRecord(primitive.type, primitive.value, source));
        }
        if (['js-token', 'js-literal', 'json-value', 'json-object-keys', 'archive-entry'].includes(primitive.type)) {
          knowledge.push(fieldRecord(primitive.type, primitive.value, source));
        }
      }

      const structure = unit.structure ?? {};
      for (const name of structure.definitions ?? []) knowledge.push(fieldRecord('definition', name, source));
      for (const name of structure.exports ?? []) knowledge.push(fieldRecord('export', name, source));
      for (const name of structure.calls ?? []) causal.push(fieldRecord('call', name, source));
      for (const name of structure.assignments ?? []) causal.push(fieldRecord('assignment', name, source));
      for (const specifier of structure.dependencies ?? []) dependency.push(fieldRecord('module', specifier, source));
      if (structure.moduleSyntax) dependency.push(fieldRecord('runtime-capability', 'module-loader', source));

      stateSpace.push(fieldRecord('artifact-state', {
        name: source,
        kind: unit.kind,
        size: unit.size,
        moduleSyntax: Boolean(structure.moduleSyntax),
      }, source));
    }

    const fields = Object.freeze({
      shared: dedupeRecords(shared),
      knowledge: dedupeRecords(knowledge),
      causal: dedupeRecords(causal),
      dependency: dedupeRecords(dependency),
      stateSpace: dedupeRecords(stateSpace),
    });

    return Object.freeze({
      schema: 'synthia.five-field-execution-map.v1',
      order: EXECUTION_FIELDS,
      fields,
      counts: Object.freeze(Object.fromEntries(EXECUTION_FIELDS.map((field) => [field, fields[field].length]))),
    });
  }

  observe(fieldMap, execution = {}) {
    const fields = {};
    for (const field of EXECUTION_FIELDS) fields[field] = [...(fieldMap.fields?.[field] ?? [])];
    const source = execution.kind ?? 'execution';
    fields.causal.push(fieldRecord('execution-outcome', {
      ok: Boolean(execution.ok),
      path: execution.path ?? execution.engine ?? null,
      error: execution.ok ? null : execution.error ?? null,
    }, source));
    fields.stateSpace.push(fieldRecord('transition', {
      from: 'compiled',
      to: execution.ok ? 'executed' : 'blocked',
      runtime: execution.result?.engine ?? execution.engine ?? execution.path ?? null,
    }, source));
    if (Array.isArray(execution.unresolvedDependencies)) {
      for (const requirement of execution.unresolvedDependencies) {
        fields.dependency.push(fieldRecord('unresolved', requirement, source));
      }
    }
    const frozen = Object.freeze(Object.fromEntries(EXECUTION_FIELDS.map((field) => [field, dedupeRecords(fields[field])])));
    return Object.freeze({
      schema: fieldMap.schema,
      order: EXECUTION_FIELDS,
      fields: frozen,
      counts: Object.freeze(Object.fromEntries(EXECUTION_FIELDS.map((field) => [field, frozen[field].length]))),
    });
  }
}

export default FiveFieldCompiler;
