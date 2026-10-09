// Presentation of existing constituent identity and mesh relations.
// Color is supplied by the address resolver, never inferred from spelling.
export function expressionMembers(world, id) {
  const particle = world.particles.get(id);
  if (particle) return [particle];
  const composite = world.composites.get(id);
  if (!composite) throw new Error(`Unknown expression: ${id}`);
  return composite.members.flatMap(member => expressionMembers(world, member));
}
export function letterState(world, id) {
  const particle = world.particles.get(id);
  if (!particle) throw new Error(`Unknown letter: ${id}`);
  const relations = world.edges.filter(edge => edge.source === id || edge.target === id);
  const separated = relations.some(edge => !edge.active);
  const strained = relations.some(edge => {
    if (!edge.active) return false;
    const a = world.particles.get(edge.source), b = world.particles.get(edge.target);
    return Math.abs(Math.hypot(b.x-a.x,b.y-a.y)-edge.rest)/edge.rest >= edge.breakStrain/2;
  });
  return { glyph: particle.qualities.glyph, state: separated ? 'separated' : strained ? 'strained' : 'connected' };
}
export function resolveExpressionColor(world, id, { color, address, dimension, provenance }) {
  if (typeof color !== 'string' || !/^#[0-9a-f]{6}$/i.test(color)) throw new Error('Resolved color must be #RRGGBB');
  if (!address || !dimension || !provenance) throw new Error('Resolved color requires address, dimension and provenance');
  const members = expressionMembers(world, id);
  const event = world.record('expression-color', { expression: id, color, address, dimension, provenance, members: members.map(p=>p.id) });
  const resolution = structuredClone({ color, address, dimension, provenance, event: event.id });
  for (const particle of members) particle.expressionColor = resolution;
  return resolution;
}

// Encode resolved RGB channels. Six hexagram lines are not automatically RGB.
export function rgbToHex(red, green, blue) {
  const channels = [red, green, blue];
  for (const value of channels) {
    if (!Number.isInteger(value) || value < 0 || value > 255) throw new RangeError('RGB channel must be an integer in 0..255');
  }
  return '#' + channels.map(value => value.toString(16).padStart(2,'0')).join('');
}
export function binaryRgbToHex(red, green, blue) {
  for (const bits of [red, green, blue]) {
    if (typeof bits !== 'string' || !/^[01]{8}$/.test(bits)) throw new Error('Each RGB channel requires exactly eight binary digits');
  }
  return rgbToHex(...[red, green, blue].map(bits => parseInt(bits,2)));
}
