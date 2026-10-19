// Neural expander — deterministic template-based expansion of grammar output
// into 2nd-person present-tense prose.
//
// LLM SEAM: `expander` is pluggable. Any object exposing
//   expand({ situation, hexagramMeaning, destinySummary }) -> Promise<string>|string
// can replace TemplateExpander (e.g. an LLM client). The game module accepts
// it via create({ expander }) / setExpander(). The default is fully offline.

export class TemplateExpander {
  /**
   * Expand a grammar SITUATION into prose.
   * @param {object} input - {situation, hexagramMeaning, destinySummary}
   * @returns {string} 2nd-person present-tense prose
   */
  expand({ situation, hexagramMeaning, destinySummary }) {
    if (!situation?.setting) throw new Error('neuralExpander: situation required');
    const s = situation;
    const ent = s.entities.items.map((e) => e.name.replaceAll('-', ' '));
    const lines = [];
    lines.push(
      `You stand where ${s.setting.elementScene.name.replaceAll('-', ' ')} holds the ground, and ${s.setting.temporalMarker}.`,
    );
    lines.push(
      `${cap(s.atmosphere.lighting)} drapes the scene in ${s.atmosphere.emotionalTone}; something in your chest answers with ${s.atmosphere.emotionalState.name}.`,
    );
    lines.push(
      `Before you: ${listJoin(ent)}. ${cap(s.conflict.description)}.`,
    );
    if (hexagramMeaning) lines.push(`The moment reads as ${hexagramMeaning}.`);
    if (destinySummary) lines.push(`Your thread: ${destinySummary}.`);
    lines.push(
      `You may touch what calls you, trace a sign in the air, join two things into one — or simply wait and let the pattern speak first.`,
    );
    return lines.join(' ');
  }
}

function cap(str) { return str.charAt(0).toUpperCase() + str.slice(1); }
function listJoin(items) {
  if (items.length <= 1) return items.join('');
  return items.slice(0, -1).join(', ') + ', and ' + items[items.length - 1];
}

export const defaultExpander = new TemplateExpander();
