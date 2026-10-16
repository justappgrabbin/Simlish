import { StateSpaceEngine } from '../donors/recovered/you-n-i-verse-corrected/state-space-engine-v2.ported.mjs';
import { expressAddress } from './address-expression.mjs';
import { DIMENSION_QUALITIES } from '../donors/dream-habitat/dimension-qualities.mjs';

export const SWARM_FIELDS = ['Movement', 'Evolution', 'Being', 'Design', 'Space'];
const qualities = node => ({ amplitude: node.amplitude, phase: node.phase });
function projection(node, observer = false) {
  const source = DIMENSION_QUALITIES[node.dimension];
  return { scale: observer ? 'observer' : 'macro/micro', qualities: qualities(node),
    macro: observer ? null : { ...qualities(node), chain: source?.macroChain ?? [] },
    micro: observer ? null : { expression: source?.micro?.value ?? null, source: source?.micro?.source ?? null },
    microStates: observer ? [] : node.children.map(child => ({ id: child.id, dimension: child.dimension, qualities: qualities(child) })) };
}

// Every unit has all five projections of the same resolved source node. Space
// observes their relationship field. Rendering never rewrites source addresses.
export function resolveRealmSwarm(participants = [], tick = 0) {
  const engine = new StateSpaceEngine();
  const units = [];
  for (const participant of participants) {
    for (const item of participant.addresses ?? []) {
      const address = item.expression.address;
      if (![address.gate, address.line, address.color, address.tone, address.base].every(Number.isFinite)) continue;
      const raw = engine.createNode(address.gate, address.line, address.color, address.tone, address.base);
      raw.id = `${participant.id}:${item.id}`;
      const seed = engine.operators.Movement.apply(raw, engine.state);
      engine.state.nodes.set(seed.id, seed);
      units.push({ participant, address, seed });
    }
  }
  for (const unit of units) unit.nodes = Object.fromEntries(SWARM_FIELDS.filter(field => field !== 'Space').map(field => [field, engine.operators[field].apply(unit.seed, engine.state)]));
  const edges = engine.state.edges.map(({ from, to, weight, coherence, type }) => ({ from, to, weight, coherence, type }));
  const pieces = units.map(({ participant, address, seed, nodes }) => {
    const space = engine.operators.Space.apply(seed, engine.state);
    const fields = Object.fromEntries(SWARM_FIELDS.map(field => [field, projection(field === 'Space' ? space : nodes[field], field === 'Space')]));
    fields.Design.relationships = edges.filter(edge => edge.from === seed.id || edge.to === seed.id);
    fields.Space.observations = fields.Design.relationships;
    return { id: seed.id, ownerId: participant.id, position: participant.position ?? [0, 1, 0],
      address, expression: expressAddress(address, { encodings: false }), fields,
      renderProjection: 'Being', amplitude: fields.Being.qualities.amplitude, phase: fields.Being.qualities.phase };
  });
  const hexagrams = Array.from({ length: 64 }, (_, i) => {
    const expression = expressAddress({ gate: i + 1 }, { encodings: false });
    return { gate: i + 1, ...expression.structure };
  });
  return { pieces, edges, hexagrams, fields: SWARM_FIELDS, tick,
    coherence: engine.calculateCoherence(), rules: 'YOU-N-I-VERSE simultaneous five-field projections', mode: 'local-mesh' };
}
