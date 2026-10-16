import test from 'node:test';
import assert from 'node:assert/strict';
import { ExecutableFormation } from './formation-spine.mjs';
import { FunctionRegistry } from './vendor/klein/index.mjs';

const address = Object.freeze({ mode: 'macro', gate: 17, line: 1, color: 1, tone: 1, base: 1 });
const singleEdge = () => ({
  nodes: [{ id: 'input' }, { id: 'output' }],
  edges: [{ id: 'increment-step', from: 'input', to: 'output', functionId: 'increment' }],
});
const twoEdges = () => ({
  nodes: [{ id: 'input' }, { id: 'middle' }, { id: 'output' }],
  edges: [
    { id: 'increment-step', from: 'input', to: 'middle', functionId: 'increment' },
    { id: 'square-step', from: 'middle', to: 'output', functionId: 'square' },
  ],
});
const makeFormation = () => {
  const registry = new FunctionRegistry();
  registry.register('increment', value => value + 1);
  registry.register('square', value => value * value);
  return new ExecutableFormation({ id: 'swarm-formation-17', address, registry, graph: singleEdge() });
};

test('addressed formation executes, restructures graph, and executes with same logical identity', async () => {
  const formation = makeFormation();
  const initial = formation.describe();
  assert.equal(initial.revisions.length, 1);
  assert.equal(formation.kernel.states.size >= 64, true);
  assert.equal((await formation.execute(2)).outputs.output, 3);

  const changed = formation.transform(twoEdges(), { reason: 'insert-composable-operation' });
  assert.equal(changed.id, initial.id);
  assert.equal(changed.revisions.length, 2);
  assert.notEqual(changed.activeRevision, initial.activeRevision);
  assert.notEqual(changed.revisions[1].artifactId, initial.revisions[0].artifactId);
  assert.equal(changed.revisions[1].graph.edges.length, 2, 'actual compiled graph structure changed');
  assert.equal((await formation.execute(2)).outputs.output, 9, 'new graph executes two registered operations');
  assert.equal((await formation.execute(2, { revisionId: initial.activeRevision })).outputs.output, 3,
    'former executable revision remains available');

  const projection = formation.kernel.project(address, 'code');
  assert.equal(projection.value.formationId, initial.id);
  assert.equal(projection.value.revisionId, changed.activeRevision);
  assert.equal(projection.value.graph.nodes.length, 3);

  const history = formation.history();
  assert.deepEqual(history.map(({ type }) => type), [
    'formation-created', 'formation-executed', 'formation-transformed',
    'formation-executed', 'formation-executed',
  ]);
  assert.deepEqual(history.map(({ sequence }) => sequence), [...history.map(({ sequence }) => sequence)].sort((a, b) => a - b));
  assert.equal(history[1].payload.outputs.output, 3);
  assert.equal(history[3].payload.outputs.output, 9);
  assert.equal(history[2].payload.previousRevision, initial.activeRevision);
});

test('invalid graph mutation is rejected without replacing active executable graph', async () => {
  const formation = makeFormation();
  const prior = formation.describe();
  const invalid = twoEdges();
  invalid.edges[1].functionId = 'not-registered';
  assert.throws(() => formation.transform(invalid), /no registered implementation/i);
  assert.equal(formation.describe().activeRevision, prior.activeRevision);
  assert.equal(formation.describe().revisions.length, 1);
  assert.equal((await formation.execute(5)).outputs.output, 6);
  assert.ok(formation.history().some(event => event.type === 'formation-transform-rejected'));
});

test('runtime failure is recorded and previous executable remains recoverable', async () => {
  const registry = new FunctionRegistry();
  registry.register('fails-for-negative', value => {
    if (value < 0) throw new Error('negative-value');
    return value * 2;
  });
  const formation = new ExecutableFormation({
    id: 'swarm-error-recovery', address, registry,
    graph: { nodes: [{ id: 'input' }, { id: 'output' }],
      edges: [{ from: 'input', to: 'output', functionId: 'fails-for-negative' }] },
  });
  await assert.rejects(() => formation.execute(-1), /negative-value/);
  assert.equal((await formation.execute(4)).outputs.output, 8);
  const outcomes = formation.history().filter(event => event.type === 'formation-executed');
  assert.deepEqual(outcomes.map(event => event.payload.status), ['failed', 'complete']);
});
