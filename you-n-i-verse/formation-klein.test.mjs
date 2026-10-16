import test from 'node:test';
import assert from 'node:assert/strict';
import { ExecutableFormation } from './formation-spine.mjs';
import { FunctionRegistry } from './vendor/klein/index.mjs';
import { AutoLingMemory, DiseminerMemory } from './vendor/klein/klein-tools.mjs';

const address = { mode: 'macro', gate: 17, line: 1, color: 1, tone: 1, base: 1 };
const edge = (from, to, functionId) => ({ from, to, functionId });

test('same executable identity becomes AutoLing and then composes AutoLing + Diseminer', async () => {
  // Native Klein operations supply semantics; ATO supplies the executable graph.
  const ling = new AutoLingMemory();
  ling.learn({
    id: 'has-property.v1', relation: 'HAS_PROPERTY',
    pattern: ['$object', 'is', '$property'], template: '$object is $property',
  });
  const diseminer = new DiseminerMemory();
  const registry = new FunctionRegistry();
  registry.register('arithmetic.double', value => value * 2);
  registry.register('autoling.recognize', input => ling.recognize(input.text));
  registry.register('diseminer.ingest-recognized', recognized => {
    if (!recognized.ok) return { ok: false, reason: 'UNRECOGNIZED' };
    const document = diseminer.ingest(recognized.text, { source: 'formed-autoling' });
    return { ok: true, document, relation: recognized.matches[0].relation };
  });

  const formation = new ExecutableFormation({
    id: 'evolving-klein-tool', address, registry,
    graph: {
      nodes: [{ id: 'input' }, { id: 'output' }],
      edges: [edge('input', 'output', 'arithmetic.double')],
    },
  });
  const arithmetic = formation.describe().activeRevision;
  assert.equal((await formation.execute(4)).outputs.output, 8);

  formation.transform({
    nodes: [{ id: 'input' }, { id: 'output' }],
    edges: [edge('input', 'output', 'autoling.recognize')],
  }, { reason: 'form-native-autoling' });
  const autoling = formation.describe().activeRevision;
  assert.equal((await formation.execute({ text: 'fox is curious' })).outputs.output.matches[0].relation, 'HAS_PROPERTY');

  formation.transform({
    nodes: [{ id: 'input' }, { id: 'recognition' }, { id: 'output' }],
    edges: [
      edge('input', 'recognition', 'autoling.recognize'),
      edge('recognition', 'output', 'diseminer.ingest-recognized'),
    ],
  }, { reason: 'compose-native-klein-tools' });
  const formed = await formation.execute({ text: 'wolf is alert' });
  assert.equal(formed.outputs.output.ok, true);
  assert.equal(formed.outputs.output.relation, 'HAS_PROPERTY');
  assert.deepEqual(formed.outputs.output.document.tokens, ['wolf', 'is', 'alert']);
  assert.equal(diseminer.documents.length, 1, 'real Disseminer memory mutated');
  assert.equal((await formation.execute(4, { revisionId: arithmetic })).outputs.output, 8);
  assert.equal((await formation.execute({ text: 'fox is curious' }, { revisionId: autoling }))
    .outputs.output.matches[0].bindings.$object, 'fox');

  const snapshot = formation.describe();
  assert.equal(snapshot.id, 'evolving-klein-tool');
  assert.equal(snapshot.revisions.length, 3);
  assert.equal(snapshot.revisions[2].graph.edges.length, 2);
  assert.equal(snapshot.revisions[2].previousRevision, autoling);
  assert.equal(formation.kernel.project(address, 'code').value.revisionId, snapshot.activeRevision);
  assert.equal(formation.history().filter(event => event.type === 'formation-transformed').length, 2);
});
